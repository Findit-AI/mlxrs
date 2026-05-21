//! DSP primitives: window family (Hann/Hamming/Blackman/Bartlett), STFT,
//! inverse STFT, mel filterbank, mel + log-mel spectrogram.
//!
//! Faithful 1:1 port of the corresponding `mlx_audio.dsp` core
//! (`hanning`, `hamming`, `blackman`, `bartlett`, `STR_TO_WINDOW_FN`, `stft`,
//! `istft`, `mel_filters`) at <https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py>.
//! Out of scope for this PR: the `ISTFTCache` batched/cached overlap-add
//! helper, Kaldi-style features, BS.1770 loudness, biquad filters, dither —
//! see [`crate::audio`] for the scope fence.
//!
//! ## API conventions
//! - Window construction is **symmetric** (`periodic=False` in `mlx-audio`):
//!   the first and last samples are zero. This matches scipy's
//!   `windows.hann(N, sym=True)` and the `mlx-audio` default for STFT. The
//!   string→window dispatch ([`window_from_name`]) mirrors `mlx-audio`'s
//!   `STR_TO_WINDOW_FN` table (`"hann"`/`"hanning"`/`"hamming"`/`"blackman"`/
//!   `"bartlett"`).
//! - STFT mirrors `mlx_audio.dsp.stft` defaults: `center=True`,
//!   `pad_mode="reflect"`. Output layout is **`(num_frames, n_fft / 2 + 1)`
//!   complex** (mlx-c `rfft` yields `Complex64` natively), as in the
//!   reference.
//! - [`istft`] inverts [`stft`] **in that same `(num_frames, n_fft / 2 + 1)`
//!   layout** (so `istft(&stft(x, ..)?, ..)` composes directly). This is a
//!   deliberate, semantics-preserving adaptation of `mlx_audio.dsp.istft`,
//!   which documents a frequency-major `(n_fft / 2 + 1, num_frames)` input
//!   and irffts along axis 0; see [`istft`] for the full rationale (the
//!   reference's `win_length` default is also derived from the frequency
//!   dimension here, fixing an axis bug in the upstream default formula).
//! - Mel filterbank uses the HTK formula
//!   (`mel = 2595 * log10(1 + hz / 700)`) and returns shape
//!   **`(n_mels, n_fft / 2 + 1)`**.
//! - `log_mel_spectrogram` uses `log(max(mel, floor))` with `floor` chosen
//!   via the [`LogFloor`] enum (default [`LogFloor::Whisper`] = `1e-10`,
//!   matching the Whisper / mlx-audio front-end). [`LogFloor::Kaldi`] =
//!   `1e-8` matches the floor literal in `mlx-audio/mlx_audio/dsp.py:950`
//!   — floor-constant parity only; the upstream mel-filterbank
//!   `get_mel_banks_kaldi` path is out of scope (see the per-variant
//!   `LogFloor::Kaldi` docs). Tracks mlx-audio's literal, NOT the
//!   upstream kaldi-asr `FbankComputer` floor of `f32::EPSILON`.

use std::f32::consts::PI;

use crate::{
  Array, Error, Result,
  ops::{
    self,
    fft::{self, FftNorm},
  },
};

/// HTK mel formula scale: `mel = MEL_HZ_DIV * log10(1 + hz / MEL_HZ_BREAK)`.
/// Matches `mlx-audio/mlx_audio/dsp.py:510` (`hz_to_mel("htk")` branch).
const MEL_HZ_DIV: f32 = 2595.0;
/// HTK mel formula break frequency (Hz). Matches `mlx-audio/mlx_audio/dsp.py:510`.
const MEL_HZ_BREAK: f32 = 700.0;
/// Log base used by both the HTK forward formula (`log10`) and the inverse
/// (`10^x`). Centralized so a future Slaney-style mel port stays consistent.
const MEL_LOG_BASE: f32 = 10.0;

/// Whisper-style log-mel floor used by `mlx-audio`'s Whisper / mlx-audio
/// front-end path (`mlx-audio/mlx_audio/dsp.py` whisper-style mel path).
const LOG_FLOOR_WHISPER: f32 = 1e-10;
/// `mlx-audio`'s "Kaldi-style" log-mel floor: the literal `1e-8` baked into
/// `mlx-audio/mlx_audio/dsp.py:950` after `get_mel_banks_kaldi`. NOTE this
/// does NOT match the upstream kaldi-asr `FbankComputer` floor of
/// `f32::EPSILON` (~`1.19e-7`) — see [`LogFloor::Kaldi`] for the rationale.
const LOG_FLOOR_KALDI: f32 = 1e-8;

/// Hard ceiling on [`istft`]'s overlap-add *work* — the number of
/// scatter/update elements `num_frames * frame_width` (`frame_width =
/// n_fft`). The OLA *output* length `t` is already capped at
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES), but with
/// small hops the scatter workload is orders of magnitude larger than `t`
/// (e.g. `num_frames=65536, n_fft=65536, hop=1` → `t≈131071` but the
/// scatter touches `4.29e9` indices). We therefore reject any
/// frame/window/hop combination whose real index count exceeds this cap
/// *before* allocating the index buffer (`try_reserve`) or building any
/// broadcast/flattened intermediate. 64 Mi-elements (256 MiB of `i32`
/// indices + matching f32 updates) is a generous ceiling that still admits
/// every realistic STFT round-trip while excluding pathological / lazily-
/// shaped inputs that would otherwise drive multi-GB allocation.
const MAX_OLA_WORK: usize = 64 * 1024 * 1024;

/// The numerical floor applied to mel energies before `log` to avoid
/// `log(0) = -inf` and to bound the dynamic range of the resulting
/// log-mel feature.
///
/// `mlx-audio` ships two distinct log-floor conventions that differ by
/// **2 orders of magnitude** with no rationale documented upstream —
/// `1e-10` in the Whisper-style front-end (deeper floor, wider dynamic
/// range) vs `1e-8` in the `get_mel_banks_kaldi` path. Mixed pipelines
/// produce subtly different features, so we expose the choice
/// explicitly rather than baking in either constant.
///
/// Defaults to [`LogFloor::Whisper`] (the mlxrs reference target;
/// preserves the previous port's behavior byte-identically).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LogFloor {
  /// `1e-10` — matches `mlx-audio`'s Whisper-style mel path.
  #[default]
  Whisper,
  /// `1e-8` — matches `mlx-audio/mlx_audio/dsp.py:950`'s literal floor
  /// (the value clamped before `log` in the `get_mel_banks_kaldi` path).
  ///
  /// **Floor-constant parity only.** This variant changes the
  /// `log(max(mel, floor))` clamp value to `1e-8`; the mel filterbank
  /// produced by [`mel_filter_bank`] is still the HTK formula (see the
  /// `# API conventions` section of this module's doc). Selecting
  /// [`LogFloor::Kaldi`] does NOT route through `get_mel_banks_kaldi`
  /// or otherwise reproduce the full kaldi-style mel pipeline (that
  /// path is out of scope for this PR per the module docs).
  ///
  /// This deliberately tracks `mlx-audio`'s `1e-8` literal — NOT the
  /// upstream kaldi-asr `FbankComputer` floor of `f32::EPSILON`
  /// (~`1.19e-7`). mlxrs is a faithful port of `mlx-audio`, so floor-
  /// constant parity for mlx-audio's two log-mel paths is the goal of
  /// this enum.
  Kaldi,
  /// A custom user-chosen floor. Useful for pipelines mixing mlx-audio's
  /// two floor choices, for floor-constant parity with upstream kaldi-asr
  /// via `LogFloor::Custom(f32::EPSILON)` (subject to the same caveat
  /// as [`LogFloor::Kaldi`]: this changes only the log clamp, not the
  /// upstream mel filterbank path), or for other reproducibility-
  /// sensitive workflows.
  ///
  /// Non-finite (`NaN`, `+/-inf`) and non-positive values (`<= 0.0`,
  /// including `-0.0`) get clamped to [`f32::MIN_POSITIVE`] inside
  /// [`LogFloor::value`] so the resulting `log(floor)` is always finite.
  Custom(f32),
}

impl LogFloor {
  /// The numeric floor value, guaranteed `> 0.0` and finite so
  /// `log(floor)` is always finite.
  pub fn value(self) -> f32 {
    match self {
      LogFloor::Whisper => LOG_FLOOR_WHISPER,
      LogFloor::Kaldi => LOG_FLOOR_KALDI,
      LogFloor::Custom(x) => {
        if x.is_finite() && x > 0.0 {
          x
        } else {
          f32::MIN_POSITIVE
        }
      }
    }
  }
}

/// Shared scaffolding for the symmetric (`periodic=False`) window family:
/// validates `n`, applies the public-input allocation cap, and materializes
/// `[sample(k) for k in 0..n]` on the CPU via a recoverable
/// `try_reserve_exact`.
///
/// `name` only flavors the error messages so each public window keeps its
/// own diagnostic prefix; `sample` receives `(k, denom)` where
/// `denom = (n - 1) as f32` (the `periodic=False` denominator shared by
/// every `mlx-audio` window). The window kinds differ ONLY in this closure,
/// so the guards / cap / fallible allocation can't drift between them.
fn symmetric_window(name: &str, n: usize, sample: impl Fn(usize, f32) -> f32) -> Result<Array> {
  if n < 2 {
    return Err(Error::Backend {
      message: format!("{name}: n must be >= 2 (got {n})"),
    });
  }
  // Cap on public-input-driven allocation — defends against an
  // adversarial / fuzzer-supplied `n = usize::MAX` that would otherwise
  // attempt a 16 EiB infallible allocation. Real-world windows are
  // typically <= a few thousand samples; 64 Mi-samples (256 MiB of f32)
  // is a generous ceiling that still excludes pathological inputs.
  if n > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "{name}: n {n} exceeds the {} cap",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }
  let n_i32 = i32::try_from(n).map_err(|_| Error::Backend {
    message: format!("{name}: n {n} exceeds i32::MAX"),
  })?;

  // Materialize on the CPU (cheap; n is bounded above) via a
  // recoverable `try_reserve_exact` so the cap above (and any
  // future allocation budget) cannot abort the host on a fuzzer input.
  let denom = (n - 1) as f32;
  let mut buf: Vec<f32> = Vec::new();
  buf.try_reserve_exact(n).map_err(|e| Error::Backend {
    message: format!("{name}: reservation for {n} elements failed: {e}"),
  })?;
  for k in 0..n {
    buf.push(sample(k, denom));
  }
  Array::from_slice::<f32>(&buf, &[n_i32])
}

/// Symmetric Hann window: `w[k] = 0.5 * (1 - cos(2π k / (n - 1)))` for
/// `k in 0..n`. The first and last samples are zero.
///
/// Matches `mlx_audio.dsp.hanning(n, periodic=False)` (the STFT default).
///
/// # Errors
/// - Returns [`Error::Backend`] when `n < 2`. The reference Python form
///   would divide by zero for `n == 1` (silently producing `NaN`); we
///   reject upfront. `n == 0` would produce an empty zero-length window
///   which is never useful for spectral analysis.
/// - Returns [`Error::Backend`] when `n` exceeds the
///   [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap or
///   `i32::MAX`, or if the backing allocation fails.
pub fn hann_window(n: usize) -> Result<Array> {
  symmetric_window("hann_window", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.5 * (1.0 - theta.cos())
  })
}

/// Symmetric Hamming window: `w[k] = 0.54 - 0.46 * cos(2π k / (n - 1))` for
/// `k in 0..n`. Endpoints are `0.08` (not zero, unlike Hann).
///
/// Matches `mlx_audio.dsp.hamming(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn hamming(n: usize) -> Result<Array> {
  symmetric_window("hamming", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.54 - 0.46 * theta.cos()
  })
}

/// Symmetric Blackman window:
/// `w[k] = 0.42 - 0.5 * cos(2π k / (n - 1)) + 0.08 * cos(4π k / (n - 1))`
/// for `k in 0..n`. Endpoints are zero (modulo f32 rounding ~`-1.4e-17`).
///
/// Matches `mlx_audio.dsp.blackman(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn blackman(n: usize) -> Result<Array> {
  symmetric_window("blackman", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos()
  })
}

/// Symmetric Bartlett (triangular) window:
/// `w[k] = 1 - 2 * |k - (n - 1) / 2| / (n - 1)` for `k in 0..n`. Rises
/// linearly to `1` at the center and back to `0` at both endpoints.
///
/// Matches `mlx_audio.dsp.bartlett(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn bartlett(n: usize) -> Result<Array> {
  symmetric_window("bartlett", n, |k, denom| {
    1.0 - 2.0 * (k as f32 - denom / 2.0).abs() / denom
  })
}

/// String → window dispatch, mirroring `mlx-audio`'s `STR_TO_WINDOW_FN`
/// table. The lookup is case-insensitive (matching the reference's
/// `window.lower()` in `stft`/`istft`):
/// - `"hann"` / `"hanning"` → [`hann_window`]
/// - `"hamming"` → [`hamming`]
/// - `"blackman"` → [`blackman`]
/// - `"bartlett"` → [`bartlett`]
///
/// All windows are the symmetric (`periodic=False`) form, as in `mlx-audio`.
///
/// # Errors
/// - [`Error::Backend`] for an unknown window name (mirrors the reference's
///   `ValueError(f"Unknown window function: {window}")`).
/// - Propagates the constructor errors of the selected window (see
///   [`hann_window`]).
pub fn window_from_name(name: &str, n: usize) -> Result<Array> {
  match name.to_ascii_lowercase().as_str() {
    "hann" | "hanning" => hann_window(n),
    "hamming" => hamming(n),
    "blackman" => blackman(n),
    "bartlett" => bartlett(n),
    other => Err(Error::Backend {
      message: format!("window_from_name: unknown window function: {other}"),
    }),
  }
}

/// Manual `reflect`-mode pad along axis 0 (1-D arrays).
///
/// `prefix = samples[1..=padding][::-1]`, `suffix =
/// samples[len-padding-1..len-1][::-1]`, then `concatenate([prefix,
/// samples, suffix])`. Matches `mlx_audio.dsp.stft._pad(..., pad_mode="reflect")`
/// byte-for-byte. mlx-c's `mlx_pad` only supports `"constant"` and `"edge"`,
/// so reflect is built from slice + concatenate here (same construction
/// the python reference uses).
///
/// # Errors
/// - [`Error::Backend`] if `padding > samples_len - 1` (not enough samples
///   to reflect — would require `samples[len-padding-1]` which underflows
///   for `padding >= len`). The reference Python form would index out of
///   bounds and return a malformed array.
fn reflect_pad_1d(samples: &Array, padding: usize) -> Result<Array> {
  if padding == 0 {
    return samples.try_clone();
  }
  let shape = samples.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!("reflect_pad_1d: expected 1-D input, got {}-D", shape.len()),
    });
  }
  let len = shape[0];
  // Need indices `samples[1..=padding]` AND `samples[len-padding-1..len-1]`
  // to exist — i.e. `len >= padding + 1`.
  if len < padding + 1 {
    return Err(Error::Backend {
      message: format!(
        "reflect_pad_1d: samples len {len} too short for reflect padding {padding} \
         (need len >= padding + 1)"
      ),
    });
  }

  let p_i32 = i32::try_from(padding).map_err(|_| Error::Backend {
    message: format!("reflect_pad_1d: padding {padding} exceeds i32::MAX"),
  })?;
  let len_i32 = i32::try_from(len).map_err(|_| Error::Backend {
    message: format!("reflect_pad_1d: samples len {len} exceeds i32::MAX"),
  })?;
  // prefix indices: `samples[padding], samples[padding-1], ..., samples[1]`.
  // `slice(start=padding, stop=0, strides=-1)` traverses `padding, padding-1,
  // ..., 1` (exclusive of `stop=0`), yielding exactly `padding` elements.
  // Boundary safe: `0` is a strictly-positive lower bound the slice never
  // reaches (the prefix never goes through index 0 — that would be a
  // double-edge reflect).
  let prefix = ops::indexing::slice(samples, &[p_i32], &[0], &[-1])?;
  // suffix indices: `samples[len-2], samples[len-3], ..., samples[len-padding-1]`,
  // exactly `padding` elements.
  //
  // mlx slice stop is exclusive of the destination, and for negative
  // strides `stop` follows mlx's `normalize_slice` rules (see
  // `mlx/ops.cpp:646` — a negative `stop` is pre-normalized by `+ n`
  // BEFORE the per-stride logic, so the post-normalize "position left of
  // 0" sentinel is `stop = -(n + 1)`, NOT `stop = -1` — `-1` would
  // post-normalize to `n - 1`).
  //
  // Two cases:
  //   1. `len - padding - 1 > 0`: traversal ends at index `len-padding-1`
  //      inclusive, so `stop = len-padding-2` (positive, the index BEFORE
  //      the last-included one).
  //   2. `len - padding - 1 == 0` (boundary: padding == len - 1): traversal
  //      must include index 0, so `stop` must post-normalize to `-1`
  //      ("position left of 0"). Using `stop = -(len + 1)` makes
  //      `e + n = -1`, exactly what mlx wants.
  let suffix_start = len_i32 - 2;
  let suffix_stop = if padding + 1 < len {
    // Inclusive-end is at index `len-padding-1 >= 1`, so the exclusive
    // stop is one less and strictly non-negative.
    len_i32 - p_i32 - 2
  } else {
    // `padding == len - 1`. Inclusive-end is index 0 — needs the
    // post-normalize-to-`-1` sentinel form (`stop = -(n + 1)`).
    //
    // Overflow note (Copilot review #3273868700): both `padding` and `len`
    // were checked to fit `i32` above via `i32::try_from`; combined with
    // `len == padding + 1` in this branch (`padding + 1 >= len` from the
    // else condition, and `len >= padding + 1` from the early check),
    // `len_i32` can be exactly `i32::MAX` (when `padding = i32::MAX - 1`,
    // `len = i32::MAX`). `len_i32 + 1` then overflows. Compute the
    // sentinel in `i64` and reject the (vanishingly rare) overflow as a
    // recoverable `Error::Backend` rather than debug-panicking / wrapping.
    let sentinel_i64 = -(i64::from(len_i32) + 1);
    i32::try_from(sentinel_i64).map_err(|_| Error::Backend {
      message: format!(
        "reflect_pad_1d: reflect-pad sentinel `-(len + 1) = {sentinel_i64}` overflows i32 \
         (len == padding + 1 == {len}, near i32::MAX boundary)"
      ),
    })?
  };
  let suffix = ops::indexing::slice(samples, &[suffix_start], &[suffix_stop], &[-1])?;
  ops::shape::concatenate(&[&prefix, samples, &suffix], 0)
}

/// Short-Time Fourier Transform along axis 0.
///
/// Faithful port of `mlx_audio.dsp.stft(x, n_fft, hop_length, win_length,
/// window="hann", center=True, pad_mode="reflect")`. The window is
/// constructed via [`hann_window`] (the only window kind in this PR;
/// hamming/blackman/bartlett are planned follow-ups). When `win_length`
/// (default = `n_fft`) is smaller than `n_fft`, the window is zero-padded
/// up to `n_fft`. `win_length > n_fft` is rejected — the reference would
/// concatenate zeros, but a longer window than the FFT length cannot occur
/// in any documented `mlx-audio` config.
///
/// Output: `(num_frames, n_fft / 2 + 1)` `Dtype::Complex64`, where
/// `num_frames = 1 + (padded_len - n_fft) / hop_length`. Matches the
/// reference layout.
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `samples` is not 1-D,
///   - `n_fft == 0`, `hop_length == 0`, or `win_length == 0`,
///   - `win_length > n_fft`,
///   - the post-pad sample count is too short to fit a single frame
///     (matches the reference's `Input is too short` raise),
///   - any size exceeds `i32::MAX`.
pub fn stft(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
) -> Result<Array> {
  if n_fft == 0 {
    return Err(Error::Backend {
      message: "stft: n_fft must be > 0".into(),
    });
  }
  if hop_length == 0 {
    return Err(Error::Backend {
      message: "stft: hop_length must be > 0".into(),
    });
  }
  let win_length = win_length.unwrap_or(n_fft);
  if win_length == 0 {
    return Err(Error::Backend {
      message: "stft: win_length must be > 0".into(),
    });
  }
  if win_length > n_fft {
    return Err(Error::Backend {
      message: format!("stft: win_length {win_length} > n_fft {n_fft} (unsupported)"),
    });
  }
  let shape = samples.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!("stft: expected 1-D input, got {}-D", shape.len()),
    });
  }

  // Window construction (hann; padded to n_fft if win_length < n_fft).
  let window = hann_window(win_length)?;
  let window = if win_length < n_fft {
    let pad_value = Array::zeros::<f32>(&[0i32; 0])?;
    let pad_axes = [0_i32];
    let pad_low = [0_i32];
    let pad_high = [
      i32::try_from(n_fft - win_length).map_err(|_| Error::Backend {
        message: format!("stft: window pad {} exceeds i32::MAX", n_fft - win_length),
      })?,
    ];
    ops::shape::pad(
      &window,
      &pad_axes,
      &pad_low,
      &pad_high,
      &pad_value,
      c"constant",
    )?
  } else {
    window
  };

  // `center=True, pad_mode="reflect"` (reference default).
  let padded = reflect_pad_1d(samples, n_fft / 2)?;
  let padded_len = padded.shape()[0];

  // Pre-frame validation: need at least one frame.
  if padded_len < n_fft {
    return Err(Error::Backend {
      message: format!(
        "stft: input is too short (padded_len={padded_len}) for n_fft={n_fft} \
         (need padded_len >= n_fft)"
      ),
    });
  }
  let num_frames = 1 + (padded_len - n_fft) / hop_length;
  if num_frames == 0 {
    return Err(Error::Backend {
      message: format!(
        "stft: input is too short for n_fft={n_fft} hop_length={hop_length} \
         (computed num_frames = 0)"
      ),
    });
  }

  // SAFETY pre-condition: the reachable element range of the strided view
  // is `(num_frames - 1) * hop_length + n_fft - 1`. We assert this is
  // strictly less than `padded_len`, so every read is in-bounds.
  let last_element_index = (num_frames - 1)
    .checked_mul(hop_length)
    .and_then(|v| v.checked_add(n_fft))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "stft: reachable element range overflows usize \
         (num_frames={num_frames}, hop_length={hop_length}, n_fft={n_fft})"
      ),
    })?;
  if last_element_index > padded_len {
    return Err(Error::Backend {
      message: format!(
        "stft: derived frame bounds {last_element_index} > padded len {padded_len} \
         (n_fft={n_fft}, hop_length={hop_length}, num_frames={num_frames}) — \
         internal invariant violated"
      ),
    });
  }
  let num_frames_i32 = i32::try_from(num_frames).map_err(|_| Error::Backend {
    message: format!("stft: num_frames {num_frames} exceeds i32::MAX"),
  })?;
  let n_fft_i32 = i32::try_from(n_fft).map_err(|_| Error::Backend {
    message: format!("stft: n_fft {n_fft} exceeds i32::MAX"),
  })?;
  let hop_i64 = i64::try_from(hop_length).map_err(|_| Error::Backend {
    message: format!("stft: hop_length {hop_length} exceeds i64::MAX"),
  })?;

  // PR #50 changed `as_strided`'s shape param to `&impl IntoShape`; an
  // array literal `&[i32; 2]` doesn't impl `IntoShape`, so we bind a
  // slice first and pass `&shape` (matching `IntoShape for &[i32]`).
  let shape: &[i32] = &[num_frames_i32, n_fft_i32];
  // SAFETY: the strided view spans element indices
  //   { i * hop_length + j  |  i in [0, num_frames),  j in [0, n_fft) }
  // The maximum reachable index is
  //   (num_frames - 1) * hop_length + (n_fft - 1) = last_element_index - 1.
  // We asserted `last_element_index <= padded_len` above, so every reachable
  // element is in `[0, padded_len)`. `padded` is row-contiguous (built via
  // concatenate of 1-D slices), so its flattened element count equals
  // `padded_len`, satisfying mlx's `as_strided` element-bounds contract.
  // `offset=0` so no out-of-front access either.
  let frames = unsafe { ops::shape::as_strided(&padded, &shape, &[hop_i64, 1], 0)? };

  // `frames * window` broadcasts the `(n_fft,)` window across each frame.
  let windowed = ops::arithmetic::multiply(&frames, &window)?;
  // rfft over the last axis (axis 1) with explicit length n_fft.
  fft::rfft(&windowed, n_fft_i32, 1, FftNorm::Backward)
}

/// Synthesis window selector for [`istft`], the idiomatic translation of
/// `mlx_audio.dsp.istft`'s `window: mx.array | str` union:
/// - [`Window::Named`] resolves a `STR_TO_WINDOW_FN` name to the **periodic**
///   form the reference uses for synthesis (`window_fn(win_length + 1)`
///   truncated to `win_length` — i.e. the symmetric window of length
///   `win_length + 1` with its trailing duplicate sample dropped). This is
///   the COLA-friendly periodic window.
/// - [`Window::Array`] supplies the synthesis window directly (the
///   reference's `else: w = window` branch). Pass the SAME window
///   [`stft`] used internally ([`hann_window`] of `win_length`) together
///   with `normalized = true` for exact `istft(stft(x))` reconstruction.
#[derive(Debug, Clone, Copy)]
pub enum Window<'a> {
  /// A `STR_TO_WINDOW_FN` name (case-insensitive). Built as the periodic
  /// window of length `win_length` (via the `win_length + 1` symmetric
  /// form, last sample dropped), matching `mlx_audio.dsp.istft`.
  Named(&'a str),
  /// A caller-supplied synthesis window array (used verbatim, then
  /// zero-padded up to `win_length` if shorter — as the reference does).
  Array(&'a Array),
}

/// Inverse Short-Time Fourier Transform — overlap-add reconstruction, the
/// inverse of [`stft`].
///
/// Faithful port of `mlx_audio.dsp.istft(x, hop_length, win_length, window,
/// center=True, length=None, normalized=False)`, adapted to mlxrs's STFT
/// layout. **`x` is `(num_frames, n_fft / 2 + 1)` `Dtype::Complex64`** — i.e.
/// exactly what [`stft`] returns — so `istft(&stft(s, ..)?, ..)` composes
/// directly. The reference instead documents a frequency-major
/// `(n_fft / 2 + 1, num_frames)` input and irffts along axis 0 then
/// transposes; here the frames are already on axis 0, so we irfft along
/// axis 1 and skip the transpose. This is a semantics-preserving adaptation,
/// not a behavior change (every sample of the reconstruction is identical to
/// the reference fed the transpose of `x`).
///
/// `win_length` defaults to `(n_freqs - 1) * 2` (= `n_fft`), derived from the
/// **frequency** dimension. The reference's documented default
/// (`(n_fft - 1) * 2`) is computed as `(x.shape[1] - 1) * 2`, which under its
/// own documented frequency-major layout reads the `num_frames` axis — an
/// upstream axis bug. We derive from `n_freqs` (`x.shape[1]` in our layout)
/// so the default equals `n_fft` and the synthesis window broadcasts against
/// the irfft output, which is the only self-consistent behavior.
/// `hop_length` defaults to `win_length / 4`.
///
/// When `win_length < n_fft`, the synthesis window is zero-padded up to the
/// full `n_fft` width exactly as [`stft`] pads its analysis window
/// (`[window(win_length), zeros(n_fft - win_length)]`), and the overlap-add
/// operates on full `n_fft`-wide irfft frames. This makes `istft` a
/// mathematically exact inverse of `stft` for `win_length < n_fft` (within
/// the COLA-covered region). The upstream reference instead slices each frame
/// down to `win_length` and trims `win_length / 2`, which is NOT a valid
/// inverse for `win_length < n_fft` (wrong center offset, dropped tail
/// energy); we deliberately implement the correct inverse here.
///
/// Normalization mirrors the reference: each output sample is divided by the
/// overlap-add sum of the (optionally squared) window. With `normalized =
/// false` the divisor is `Σ w` (simple window normalization); with
/// `normalized = true` it is `Σ w²` (COLA / `torch.istft` convention).
/// Positions whose window-sum is `<= 1e-10` are left unnormalized (matching
/// the reference's `mx.where(window_sum > 1e-10, ...)` guard). Samples outside
/// the window's COLA-covered span (e.g. the very last samples when
/// `win_length < n_fft` and the zero-padded window tail leaves them
/// uncovered) therefore stay un-normalized — this is intrinsic to the
/// windowing, not a defect.
///
/// Center / `length` ordering matches librosa / mlx-audio center semantics:
/// when `center = true` (the default), the `n_fft / 2` reflect-pad [`stft`]
/// added is removed FIRST (the centered signal begins at raw OLA index
/// `n_fft / 2`), and only then is `length` applied:
/// - `center = true,  length = None`    → the centered signal
///   `reconstructed[n_fft/2 .. t - n_fft/2]`.
/// - `center = true,  length = Some(n)` → `reconstructed[n_fft/2 .. n_fft/2 + n]`
///   (the first `n` real samples after dropping the reflected prefix).
/// - `center = false, length = Some(n)` → `reconstructed[0 .. n]`.
/// - `center = false, length = None`    → the full raw overlap-add.
///
/// Returns the reconstructed 1-D real signal (`Dtype::F32`).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `x` is not 2-D, or `n_freqs < 2` (need at least 2 bins to define
///     `n_fft = (n_freqs - 1) * 2`),
///   - `num_frames == 0`,
///   - `hop_length == 0` or `win_length == 0`,
///   - `win_length > n_fft` (the synthesis window cannot be longer than the
///     irfft frame — the reference would broadcast-fail),
///   - an explicit [`Window::Array`] is not 1-D,
///   - any derived size overflows `usize`/`i32`, the OLA output length `t`
///     exceeds the
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap, or
///     the real scatter workload `num_frames * n_fft` exceeds an internal
///     work cap (`MAX_OLA_WORK`, checked before any allocation/broadcast —
///     guards against small-hop combinations whose work explodes far past
///     `t`),
///   - the `length` trim is out of range (with `center = true`,
///     `n_fft/2 + length > t`; with `center = false`, `length > t`).
/// - Propagates window-construction errors from [`window_from_name`].
pub fn istft(
  x: &Array,
  hop_length: Option<usize>,
  win_length: Option<usize>,
  window: Window<'_>,
  center: bool,
  length: Option<usize>,
  normalized: bool,
) -> Result<Array> {
  let shape = x.shape();
  if shape.len() != 2 {
    return Err(Error::Backend {
      message: format!(
        "istft: expected 2-D (num_frames, n_freqs) input, got {}-D",
        shape.len()
      ),
    });
  }
  let num_frames = shape[0];
  let n_freqs = shape[1];
  if n_freqs < 2 {
    return Err(Error::Backend {
      message: format!("istft: n_freqs {n_freqs} < 2 (need >= 2 bins for irfft)"),
    });
  }
  if num_frames == 0 {
    return Err(Error::Backend {
      message: "istft: num_frames must be > 0".into(),
    });
  }
  // n_fft = (n_freqs - 1) * 2, the irfft target length.
  let n_fft = (n_freqs - 1).checked_mul(2).ok_or_else(|| Error::Backend {
    message: format!("istft: n_fft = (n_freqs - 1) * 2 overflows usize (n_freqs={n_freqs})"),
  })?;
  let win_length = win_length.unwrap_or(n_fft);
  if win_length == 0 {
    return Err(Error::Backend {
      message: "istft: win_length must be > 0".into(),
    });
  }
  if win_length > n_fft {
    return Err(Error::Backend {
      message: format!("istft: win_length {win_length} > n_fft {n_fft} (unsupported)"),
    });
  }
  let hop_length = hop_length.unwrap_or(win_length / 4);
  if hop_length == 0 {
    return Err(Error::Backend {
      message: "istft: hop_length must be > 0".into(),
    });
  }
  let win_length_i32 = i32::try_from(win_length).map_err(|_| Error::Backend {
    message: format!("istft: win_length {win_length} exceeds i32::MAX"),
  })?;

  // Synthesis window. Named → periodic form (symmetric of `win_length + 1`,
  // trailing sample dropped); Array → used verbatim. Either way, zero-pad up
  // to `win_length` if shorter (matches the reference).
  let window = match window {
    Window::Named(name) => {
      let win_len_p1 = win_length.checked_add(1).ok_or_else(|| Error::Backend {
        message: format!("istft: win_length {win_length} + 1 overflows usize"),
      })?;
      let full = window_from_name(name, win_len_p1)?;
      // Drop the trailing duplicate sample: full[0 .. win_length].
      ops::indexing::slice(&full, &[0], &[win_length_i32], &[1])?
    }
    Window::Array(w) => {
      if w.ndim() != 1 {
        return Err(Error::Backend {
          message: format!("istft: explicit window must be 1-D, got {}-D", w.ndim()),
        });
      }
      w.try_clone()?
    }
  };
  // Synthesis window must be padded to the **full `n_fft`** width exactly as
  // [`stft`] pads its analysis window (zeros appended at the END): `stft`
  // builds `w = [window(win_length), zeros(n_fft - win_length)]` and applies
  // it to every `n_fft`-wide frame, so a mathematically exact inverse must
  // overlap-add `n_fft`-wide frames against the *same* `n_fft`-wide window.
  // The previous code instead sliced each irfft frame down to `win_length`
  // and trimmed by `win_length / 2`, which is NOT the inverse of `stft` for
  // `win_length < n_fft` (wrong offset / dropped energy) — see the Codex
  // review note. We pad in two steps to keep the diagnostics specific:
  //   1. up to `win_length` (no-op for the Named path; pads a short explicit
  //      `Window::Array` — mirrors the reference's `w` zero-pad), then
  //   2. up to `n_fft` (the `stft` analysis-window pad).
  let w_len = window.shape()[0];
  let window = if w_len < win_length {
    let pad_value = Array::zeros::<f32>(&[0i32; 0])?;
    let pad_high = [
      i32::try_from(win_length - w_len).map_err(|_| Error::Backend {
        message: format!("istft: window pad {} exceeds i32::MAX", win_length - w_len),
      })?,
    ];
    ops::shape::pad(
      &window,
      &[0_i32],
      &[0_i32],
      &pad_high,
      &pad_value,
      c"constant",
    )?
  } else {
    window
  };
  // Pad the (win_length-wide) synthesis window up to n_fft with trailing
  // zeros, matching `stft`'s `[window, zeros(n_fft - win_length)]`. No-op
  // when win_length == n_fft (the default).
  let window = if win_length < n_fft {
    let pad_value = Array::zeros::<f32>(&[0i32; 0])?;
    let pad_high = [
      i32::try_from(n_fft - win_length).map_err(|_| Error::Backend {
        message: format!(
          "istft: window pad-to-n_fft {} exceeds i32::MAX",
          n_fft - win_length
        ),
      })?,
    ];
    ops::shape::pad(
      &window,
      &[0_i32],
      &[0_i32],
      &pad_high,
      &pad_value,
      c"constant",
    )?
  } else {
    window
  };

  // Every frame is `n_fft` wide (the irfft output width and the padded
  // synthesis-window width), so the overlap-add stride/frame width is `n_fft`.
  let frame_width = n_fft;

  // Output / window-sum buffer length: `t = (num_frames - 1) * hop + n_fft`.
  let t = (num_frames - 1)
    .checked_mul(hop_length)
    .and_then(|v| v.checked_add(frame_width))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "istft: OLA length (num_frames-1)*hop + n_fft overflows usize \
         (num_frames={num_frames}, hop={hop_length}, n_fft={n_fft})"
      ),
    })?;
  if t > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "istft: OLA length {t} exceeds the {} cap",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }

  // OOM guard on the *real* scatter/update workload (`num_frames *
  // frame_width`), checked BEFORE any broadcast / flatten / `try_reserve`.
  // The `t` cap above bounds the *output* length, but with small hops the
  // scatter touches far more elements than `t` (e.g. num_frames=65536,
  // n_fft=65536, hop=1 → t≈131071 but idx_len≈4.29e9). Reject overflow,
  // `> i32::MAX`, and `> MAX_OLA_WORK` here so a shaped/lazy input can never
  // drive a multi-GB allocation downstream.
  let idx_len = num_frames
    .checked_mul(frame_width)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "istft: scatter work count num_frames * n_fft overflows usize \
         (num_frames={num_frames}, n_fft={n_fft})"
      ),
    })?;
  if idx_len > MAX_OLA_WORK {
    return Err(Error::Backend {
      message: format!(
        "istft: scatter work count {idx_len} (num_frames={num_frames} * n_fft={n_fft}) \
         exceeds the {MAX_OLA_WORK} work cap"
      ),
    });
  }
  let idx_len_i32 = i32::try_from(idx_len).map_err(|_| Error::Backend {
    message: format!("istft: scatter work count {idx_len} exceeds i32::MAX"),
  })?;

  let t_i32 = i32::try_from(t).map_err(|_| Error::Backend {
    message: format!("istft: OLA length {t} exceeds i32::MAX"),
  })?;
  let n_fft_i32 = i32::try_from(n_fft).map_err(|_| Error::Backend {
    message: format!("istft: n_fft {n_fft} exceeds i32::MAX"),
  })?;

  // Inverse FFT of every frame along the frequency axis (axis 1):
  // (num_frames, n_freqs) complex → (num_frames, n_fft) real. Frames stay
  // full `n_fft` wide — they are NOT sliced to `win_length` (that was the
  // pre-fix bug); the trailing-zero region of the padded synthesis window
  // zeroes out the unused tail during the multiply below.
  let frames_time = fft::irfft(x, n_fft_i32, 1, FftNorm::Backward)?;

  // updates_reconstructed = (frames_time * w).flatten() — shape
  // (num_frames * n_fft,). `w` is (n_fft,) and broadcasts across the frame
  // axis.
  let windowed = ops::arithmetic::multiply(&frames_time, &window)?;
  let updates_reconstructed = ops::shape::flatten(&windowed, 0, -1)?;

  // window_norm = w*w if normalized else w; tiled across frames then flattened.
  let window_norm = if normalized {
    ops::arithmetic::multiply(&window, &window)?
  } else {
    window
  };
  // tile(window_norm, num_frames): (n_fft,) → (num_frames, n_fft).
  let window_norm_row = ops::shape::reshape(&window_norm, &(1usize, frame_width))?;
  let window_norm_tiled = ops::shape::broadcast_to(&window_norm_row, &(num_frames, frame_width))?;
  let updates_window = ops::shape::flatten(&window_norm_tiled, 0, -1)?;

  // Overlap-add destination indices:
  // indices[m, j] = m * hop + j, flattened to (num_frames * n_fft,).
  // Built CPU-side (bounded by the work cap above) as i32 — the reference
  // builds the same via arange broadcasts.
  let mut idx_buf: Vec<i32> = Vec::new();
  idx_buf
    .try_reserve_exact(idx_len)
    .map_err(|e| Error::Backend {
      message: format!("istft: index reservation for {idx_len} elements failed: {e}"),
    })?;
  let frame_width_i32 = i32::try_from(frame_width).map_err(|_| Error::Backend {
    message: format!("istft: n_fft {frame_width} exceeds i32::MAX"),
  })?;
  for m in 0..num_frames {
    // `m * hop_length < t <= i32::MAX` (t bounded above), and `+ j` stays
    // `< t`, so every index fits i32 without a per-element checked cast.
    let off = (m * hop_length) as i32;
    for j in 0..frame_width_i32 {
      idx_buf.push(off + j);
    }
  }
  let indices = Array::from_slice::<i32>(&idx_buf, &[idx_len_i32])?;

  // reconstructed / window_sum via scatter-add into zero buffers (axis 0).
  let zeros_recon = Array::zeros::<f32>(&[t_i32])?;
  let zeros_wsum = Array::zeros::<f32>(&[t_i32])?;
  let reconstructed =
    ops::indexing::scatter_add_axis(&zeros_recon, &indices, &updates_reconstructed, 0)?;
  let window_sum = ops::indexing::scatter_add_axis(&zeros_wsum, &indices, &updates_window, 0)?;

  // normalize by the (squared) window-sum where it exceeds 1e-10, else leave
  // the raw overlap-add (matches the reference's `mx.where` guard).
  let threshold = Array::full::<f32>(&[0i32; 0], 1e-10)?;
  let mask = ops::comparison::greater(&window_sum, &threshold)?;
  let normalized_recon = ops::arithmetic::divide(&reconstructed, &window_sum)?;
  let reconstructed = ops::logical::select(&mask, &normalized_recon, &reconstructed)?;

  // Final trimming. The center reflect-pad `stft` added is `n_fft / 2` on
  // EACH side (`reflect_pad_1d(samples, n_fft / 2)`), so the centered signal
  // begins at raw OLA index `pad = n_fft / 2` — NOT `win_length / 2` (the
  // old, wrong offset for `win_length < n_fft`) and the center pad must be
  // removed BEFORE `length` is applied (librosa / mlx-audio center
  // semantics):
  //   * `center == true,  length = Some(n)` → `reconstructed[pad .. pad + n]`
  //     (drop the reflected prefix, then keep `n` real samples). The pre-fix
  //     code returned `reconstructed[0 .. n]`, i.e. the reflected prefix plus
  //     a truncated head — the Codex `length` finding.
  //   * `center == true,  length = None`    → `reconstructed[pad .. t - pad]`
  //     (the centered signal; symmetric un-pad). The pre-fix `length = None`
  //     path returned `(num_frames - 1) * hop` samples and silently shortened
  //     non-hop-aligned inputs.
  //   * `center == false, length = Some(n)` → `reconstructed[0 .. n]`
  //     (no pad was added, so no offset).
  //   * `center == false, length = None`    → the full raw OLA.
  let pad = n_fft / 2;
  match (center, length) {
    (true, Some(len)) => {
      // `reconstructed[pad .. pad + len]`. Require `pad + len <= t` so the
      // requested window stays inside the reconstruction.
      let end = pad.checked_add(len).ok_or_else(|| Error::Backend {
        message: format!("istft: center offset {pad} + length {len} overflows usize"),
      })?;
      if end > t {
        return Err(Error::Backend {
          message: format!(
            "istft: center offset {pad} + length {len} = {end} exceeds reconstruction length {t}"
          ),
        });
      }
      let start = i32::try_from(pad).map_err(|_| Error::Backend {
        message: format!("istft: center trim start {pad} exceeds i32::MAX"),
      })?;
      let stop = i32::try_from(end).map_err(|_| Error::Backend {
        message: format!("istft: center trim stop {end} exceeds i32::MAX"),
      })?;
      ops::indexing::slice(&reconstructed, &[start], &[stop], &[1])
    }
    (true, None) => {
      // `reconstructed[pad .. t - pad]`. `t = (num_frames - 1) * hop + n_fft
      // >= n_fft >= 2 * (n_fft / 2) = 2 * pad`, so `t - pad >= pad` and the
      // slice is non-empty / well-ordered.
      let start = i32::try_from(pad).map_err(|_| Error::Backend {
        message: format!("istft: center trim start {pad} exceeds i32::MAX"),
      })?;
      let stop = i32::try_from(t - pad).map_err(|_| Error::Backend {
        message: format!("istft: center trim stop {} exceeds i32::MAX", t - pad),
      })?;
      ops::indexing::slice(&reconstructed, &[start], &[stop], &[1])
    }
    (false, Some(len)) => {
      if len > t {
        return Err(Error::Backend {
          message: format!("istft: requested length {len} exceeds reconstruction length {t}"),
        });
      }
      let len_i32 = i32::try_from(len).map_err(|_| Error::Backend {
        message: format!("istft: length {len} exceeds i32::MAX"),
      })?;
      ops::indexing::slice(&reconstructed, &[0], &[len_i32], &[1])
    }
    (false, None) => Ok(reconstructed),
  }
}

/// HTK mel scale: `mel = 2595 * log10(1 + hz / 700)`.
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
  MEL_HZ_DIV * (1.0 + hz / MEL_HZ_BREAK).log10()
}

/// Inverse HTK mel scale: `hz = 700 * (10^(mel / 2595) - 1)`.
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
  MEL_HZ_BREAK * (MEL_LOG_BASE.powf(mel / MEL_HZ_DIV) - 1.0)
}

/// Triangular mel filterbank matrix of shape `(n_mels, n_fft / 2 + 1)`.
///
/// Faithful port of `mlx_audio.dsp.mel_filters(sample_rate, n_fft, n_mels,
/// f_min, f_max, norm=None, mel_scale="htk")` — the HTK formula only;
/// Slaney normalization is a planned follow-up.
///
/// `f_max` defaults to `sample_rate / 2` (Nyquist) when `None`. The reference
/// builds frequency points via `mx.linspace(0, sample_rate // 2, n_freqs)`
/// which integer-divides the Nyquist — we mirror that exactly (using
/// `sample_rate as f32 / 2.0` would drift by 0.5 for odd sample rates).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `n_fft == 0`,
///   - `n_mels == 0` (no filters requested),
///   - `f_min < 0` or `f_max <= f_min`,
///   - any size exceeds `i32::MAX`.
pub fn mel_filter_bank(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  if n_fft == 0 {
    return Err(Error::Backend {
      message: "mel_filter_bank: n_fft must be > 0".into(),
    });
  }
  if n_mels == 0 {
    return Err(Error::Backend {
      message: "mel_filter_bank: n_mels must be > 0".into(),
    });
  }
  if sample_rate == 0 {
    return Err(Error::Backend {
      message: "mel_filter_bank: sample_rate must be > 0".into(),
    });
  }
  let f_max = f_max.unwrap_or((sample_rate / 2) as f32);
  if !(f_min >= 0.0 && f_max > f_min) {
    return Err(Error::Backend {
      message: format!("mel_filter_bank: invalid f_min={f_min} / f_max={f_max}"),
    });
  }

  // `n_freqs = n_fft / 2 + 1`; `n_fft / 2 <= usize::MAX / 2`, so `+ 1`
  // cannot overflow `usize`. Bound on i32 happens after the multiplication
  // check below.
  let n_freqs = n_fft / 2 + 1;
  // `n_pts = n_mels + 2`; check for overflow on `n_mels = usize::MAX` /
  // `usize::MAX - 1` before we walk `0..n_pts`.
  let n_pts = n_mels.checked_add(2).ok_or_else(|| Error::Backend {
    message: format!("mel_filter_bank: n_mels {n_mels} + 2 overflows usize"),
  })?;
  // Bank size: `n_mels * n_freqs`. The reference uses an mlx broadcast
  // graph; we materialize one `Vec<f32>` of the same logical size, so we
  // must reject any combination that would attempt a multi-GB allocation
  // (the python form would silently swap or OOM-kill).
  let bank_len = n_mels.checked_mul(n_freqs).ok_or_else(|| Error::Backend {
    message: format!(
      "mel_filter_bank: n_mels * n_freqs overflows usize \
       (n_mels={n_mels}, n_freqs={n_freqs})"
    ),
  })?;
  // i32 bounds on the final mlx shape go here, BEFORE any large allocation.
  let n_mels_i32 = i32::try_from(n_mels).map_err(|_| Error::Backend {
    message: format!("mel_filter_bank: n_mels {n_mels} exceeds i32::MAX"),
  })?;
  let n_freqs_i32 = i32::try_from(n_freqs).map_err(|_| Error::Backend {
    message: format!("mel_filter_bank: n_freqs {n_freqs} exceeds i32::MAX"),
  })?;

  // `all_freqs[i] = i * (sample_rate / 2) / (n_freqs - 1)` for the python
  // `mx.linspace(0, sample_rate // 2, n_freqs)` form. Build CPU-side;
  // n_freqs is small for any reasonable n_fft (e.g. 201 for n_fft=400).
  // Use `try_reserve_exact` for the same reason as `bank` below — a
  // crafted n_fft can drive n_freqs into multi-GB territory.
  let nyq = (sample_rate / 2) as f32;
  let denom = (n_freqs as f32 - 1.0).max(1.0);
  let mut all_freqs: Vec<f32> = Vec::new();
  all_freqs
    .try_reserve_exact(n_freqs)
    .map_err(|e| Error::Backend {
      message: format!("mel_filter_bank: reservation for n_freqs={n_freqs} failed: {e}"),
    })?;
  for i in 0..n_freqs {
    all_freqs.push(i as f32 * nyq / denom);
  }

  // Mel grid: `n_mels + 2` points (the +2 give the outer triangle edges).
  let m_min = hz_to_mel(f_min);
  let m_max = hz_to_mel(f_max);
  let m_denom = (n_pts as f32 - 1.0).max(1.0);
  let mut f_pts: Vec<f32> = Vec::new();
  f_pts.try_reserve_exact(n_pts).map_err(|e| Error::Backend {
    message: format!("mel_filter_bank: reservation for n_pts={n_pts} failed: {e}"),
  })?;
  for i in 0..n_pts {
    let m = m_min + (m_max - m_min) * (i as f32) / m_denom;
    f_pts.push(mel_to_hz(m));
  }

  // Build the filterbank directly on the CPU as `(n_mels, n_freqs)` to
  // avoid the reference's allocation chain (linspace + 4 broadcast ops);
  // this is the only place we elide an mlx-graph step in this PR — the
  // mel filter is a one-shot constant matrix per `(sample_rate, n_fft,
  // n_mels)` triple, and the on-device construction has no perf benefit.
  // Logged in docs/rust-golden-standard-followups.md (AUDIO-2).
  //
  // Use `try_reserve_exact` so a multi-GB request from a forged input
  // returns a recoverable `Error::Backend` rather than aborting on the
  // allocator's OOM panic (Rust's default behavior is to abort, not
  // unwind, on allocation failure — `Vec::with_capacity` and `vec![]`
  // share that abort path).
  let mut bank: Vec<f32> = Vec::new();
  bank
    .try_reserve_exact(bank_len)
    .map_err(|e| Error::Backend {
      message: format!("mel_filter_bank: allocation of {bank_len} f32 elements failed: {e}"),
    })?;
  bank.resize(bank_len, 0.0);
  for m in 0..n_mels {
    let left = f_pts[m];
    let center = f_pts[m + 1];
    let right = f_pts[m + 2];
    let lc = center - left;
    let cr = right - center;
    // Guard against zero-width triangles (collapsed mel bins). The
    // reference would NaN/inf on the division; we keep the bin at zero.
    if lc <= 0.0 || cr <= 0.0 {
      continue;
    }
    for (f, &freq) in all_freqs.iter().enumerate() {
      let up = (freq - left) / lc;
      let down = (right - freq) / cr;
      let v = up.min(down).max(0.0);
      bank[m * n_freqs + f] = v;
    }
  }

  Array::from_slice::<f32>(&bank, &[n_mels_i32, n_freqs_i32])
}

/// Mel spectrogram: `mel_bank @ |stft(samples)|^2`.
///
/// Returns shape `(n_mels, num_frames)` `Dtype::F32`. Combines [`stft`],
/// magnitude-squared, and [`mel_filter_bank`] in the canonical Whisper /
/// mlx-audio order.
///
/// # Errors
/// Propagates from [`stft`] and [`mel_filter_bank`].
#[allow(clippy::too_many_arguments)]
pub fn mel_spectrogram(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  n_mels: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  let spec = stft(samples, n_fft, hop_length, win_length)?;
  // `|stft|^2` — `abs` of Complex64 yields F32 magnitudes, then square.
  let mag = spec.abs()?;
  let power = mag.square()?;
  // `power` is `(num_frames, n_freqs)`; mel is `(n_mels, n_freqs)`.
  // Mel-spec layout in mlx-audio / Whisper is `(n_mels, num_frames)` =
  // `mel @ power.T`.
  let mel = mel_filter_bank(n_mels, n_fft, sample_rate, f_min, f_max)?;
  let power_t = power.transpose()?;
  ops::linalg_basic::matmul(&mel, &power_t)
}

/// Log-mel spectrogram: `log(max(mel_spectrogram, floor))` with `floor =
/// [`LogFloor::default`]` (= `1e-10`, Whisper / mlx-audio convention).
///
/// Thin forward to [`log_mel_spectrogram_with`] with the default floor —
/// output is byte-identical to the pre-`LogFloor` behavior. Use
/// [`log_mel_spectrogram_with`] to pick a different floor explicitly.
///
/// # Errors
/// Propagates from [`mel_spectrogram`].
#[allow(clippy::too_many_arguments)]
pub fn log_mel_spectrogram(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  n_mels: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  log_mel_spectrogram_with(
    samples,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    sample_rate,
    f_min,
    f_max,
    LogFloor::default(),
  )
}

/// Log-mel spectrogram with an explicit log floor — `log(max(mel, floor.value()))`.
///
/// Lets the caller pick between [`LogFloor::Whisper`] (`1e-10`, the default
/// matching the mlx-audio Whisper-style front-end), [`LogFloor::Kaldi`]
/// (`1e-8`, matching the floor literal in `mlx-audio/mlx_audio/dsp.py:950`),
/// or [`LogFloor::Custom`] for downstream reproducibility-sensitive
/// workflows. See [`LogFloor`] for the rationale and the floor-constant-
/// only scope (the mel filterbank stays the HTK one — `LogFloor::Kaldi`
/// does NOT swap in `get_mel_banks_kaldi`).
///
/// # Errors
/// Propagates from [`mel_spectrogram`].
#[allow(clippy::too_many_arguments)]
pub fn log_mel_spectrogram_with(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  n_mels: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  floor: LogFloor,
) -> Result<Array> {
  let mel = mel_spectrogram(
    samples,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    sample_rate,
    f_min,
    f_max,
  )?;
  // `maximum(mel, floor)` then `log`. Build the floor as a 0-D scalar so
  // it broadcasts against `mel`'s `(n_mels, num_frames)` shape.
  let eps = Array::full::<f32>(&[0i32; 0], floor.value())?;
  let floored = ops::arithmetic::maximum(&mel, &eps)?;
  floored.log()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Absolute tolerance for the closed-form window value checks. The
  /// formulas are evaluated in f32 here and in `mlx-audio` in f64 then cast
  /// to f32, so a few ULPs of slack is expected.
  const WIN_TOL: f32 = 1e-6;

  fn to_vec(a: &Array) -> Vec<f32> {
    // Tests own their arrays; clone so the accessor's `&mut self` (which
    // triggers the explicit eval) doesn't force a `mut` binding on callers.
    a.try_clone().unwrap().to_vec::<f32>().unwrap()
  }

  // ---- A2: window family closed-form parity (hand-derived) ----------------

  #[test]
  fn hamming_matches_closed_form_n5() {
    // 0.54 - 0.46 cos(2π k / 4) for k in 0..5:
    // k=0: 0.54-0.46 = 0.08; k=1: 0.54-0; wait cos(π/2)=0 → 0.54; k=2:
    // cos(π)=-1 → 1.0; k=3: 0.54; k=4: 0.08.
    let v = to_vec(&hamming(5).unwrap());
    let expected = [0.08_f32, 0.54, 1.0, 0.54, 0.08];
    for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "hamming[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn hamming_endpoints_are_0_08() {
    // Distinguishing feature vs Hann: Hamming endpoints are 0.08, not 0.
    let v = to_vec(&hamming(8).unwrap());
    assert!((v[0] - 0.08).abs() < WIN_TOL, "first: {}", v[0]);
    assert!((v[7] - 0.08).abs() < WIN_TOL, "last: {}", v[7]);
  }

  #[test]
  fn blackman_matches_closed_form_n5() {
    // 0.42 - 0.5 cos(2π k/4) + 0.08 cos(4π k/4):
    // k=0: 0.42-0.5+0.08 = 0.0; k=1: 0.42-0+(-0.08)=0.34; k=2:
    // 0.42+0.5+0.08=1.0; k=3: 0.34; k=4: 0.0.
    let v = to_vec(&blackman(5).unwrap());
    let expected = [0.0_f32, 0.34, 1.0, 0.34, 0.0];
    for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "blackman[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn bartlett_matches_closed_form_n5_and_n4() {
    // n=5 (odd): triangle peaking at 1.0 in the center, 0 at the ends.
    let v5 = to_vec(&bartlett(5).unwrap());
    let e5 = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
    for (i, (g, e)) in v5.iter().zip(e5.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "bartlett5[{i}]: got {g}, want {e}");
    }
    // n=4 (even): 1 - 2|k - 1.5|/3 → [0, 2/3, 2/3, 0].
    let v4 = to_vec(&bartlett(4).unwrap());
    let e4 = [0.0_f32, 2.0 / 3.0, 2.0 / 3.0, 0.0];
    for (i, (g, e)) in v4.iter().zip(e4.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "bartlett4[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn windows_reject_n_lt_2() {
    for r in [
      hamming(0),
      hamming(1),
      blackman(1),
      bartlett(0),
      bartlett(1),
    ] {
      assert!(matches!(r, Err(Error::Backend { .. })));
    }
  }

  #[test]
  fn window_from_name_dispatches_case_insensitively() {
    // "hann"/"hanning" → Hann (endpoints 0); "HAMMING" → Hamming
    // (endpoints 0.08); names are lowercased like the reference.
    let hann = to_vec(&window_from_name("HaNn", 8).unwrap());
    assert!(hann[0].abs() < WIN_TOL && hann[7].abs() < WIN_TOL);
    let hanning = to_vec(&window_from_name("hanning", 8).unwrap());
    assert_eq!(hann, hanning, "hann and hanning must be identical");
    let hamming = to_vec(&window_from_name("HAMMING", 8).unwrap());
    assert!((hamming[0] - 0.08).abs() < WIN_TOL);
    let bartlett = to_vec(&window_from_name("Bartlett", 5).unwrap());
    assert!((bartlett[2] - 1.0).abs() < WIN_TOL);
  }

  #[test]
  fn window_from_name_rejects_unknown() {
    assert!(matches!(
      window_from_name("kaiser", 8),
      Err(Error::Backend { .. })
    ));
  }

  // ---- A1: istft ----------------------------------------------------------

  /// The 16-sample test signal used for the round-trip (arbitrary but fixed).
  fn signal_16() -> [f32; 16] {
    [
      0.1, 0.5, -0.3, 0.8, -0.2, 0.6, 0.0, -0.7, 0.4, 0.9, -0.5, 0.2, 0.3, -0.1, 0.7, -0.4,
    ]
  }

  #[test]
  fn istft_reconstructs_stft_with_matching_window() {
    // Perfect reconstruction: feed istft the SAME symmetric Hann window stft
    // uses internally (via Window::Array) with normalized=true (COLA / Σw²
    // normalization), so each frame contributes `w² · x` and the divisor is
    // `Σ w²` → exact `x` wherever the window-sum is non-zero. With n_fft=8,
    // hop=4 (50% overlap) and 16 samples, the center-trim recovers the
    // original length exactly. Verified against a numpy reference to 1e-16
    // (f64); we assert 1e-5 for the f32 backend.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, None).unwrap();
    assert_eq!(spec.shape(), vec![5, 5]); // (num_frames, n_fft/2+1)

    let w = hann_window(8).unwrap();
    let rec = istft(
      &spec,
      Some(4), // hop_length (matches stft)
      Some(8), // win_length == n_fft
      Window::Array(&w),
      true, // center (undo stft's reflect pad)
      None, // length (None → center-trim)
      true, // normalized (Σw²)
    )
    .unwrap();
    let r = to_vec(&rec);
    assert_eq!(r.len(), buf.len(), "round-trip length mismatch");
    for (i, (g, e)) in r.iter().zip(buf.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "reconstruction[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  #[test]
  fn istft_center_length_removes_pad_before_truncating() {
    // Codex `length` finding: with `center = true` and an explicit `length`,
    // the center reflect-pad (`n_fft / 2 = 4`) must be removed BEFORE the
    // length cut, so the result is `reconstructed[pad .. pad + length]` — the
    // first `length` REAL samples — NOT `reconstructed[0 .. length]` (which
    // would start in the reflected prefix). We therefore expect the first 10
    // samples of the ORIGINAL signal, value-for-value. (n_fft=8, hop=4,
    // win_length=8, t = (5-1)*4 + 8 = 24.)
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, None).unwrap();
    let w = hann_window(8).unwrap();
    let rec = istft(
      &spec,
      Some(4),
      Some(8),
      Window::Array(&w),
      true,     // center: remove the n_fft/2 pad first
      Some(10), // length: keep 10 real samples after the pad
      true,     // normalized (Σw²)
    )
    .unwrap();
    let r = to_vec(&rec);
    assert_eq!(
      r.len(),
      10,
      "length override should yield exactly 10 samples"
    );
    // Must equal the first 10 ORIGINAL samples, not the reflected prefix.
    for (i, (g, e)) in r.iter().zip(buf.iter().take(10)).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "center+length reconstruction[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
    // Without center, the same `length` returns the RAW OLA head (which begins
    // with the reflected prefix), so element 0 differs from the original — a
    // direct check that the center-pad removal is what produces the real head.
    let rec_no_center = istft(
      &spec,
      Some(4),
      Some(8),
      Window::Array(&w),
      false, // center=false: no pad removal
      Some(10),
      true,
    )
    .unwrap();
    let r_nc = to_vec(&rec_no_center);
    assert_eq!(r_nc.len(), 10);
    assert!(
      (r_nc[0] - buf[0]).abs() > 1e-3,
      "center=false head should be the reflected prefix, not the original \
       (got {} vs original {})",
      r_nc[0],
      buf[0]
    );
  }

  #[test]
  fn istft_named_window_runs_and_is_finite() {
    // The Named path builds the periodic Hann (hann(win_length+1)[:-1]); this
    // is NOT identical to stft's symmetric analysis window, so it won't
    // reconstruct exactly, but it must run end-to-end and stay finite.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, None).unwrap();
    let rec = istft(
      &spec,
      Some(4),
      Some(8),
      Window::Named("hann"),
      true,
      None,
      false,
    )
    .unwrap();
    for (i, v) in to_vec(&rec).iter().enumerate() {
      assert!(v.is_finite(), "istft[{i}] not finite: {v}");
    }
  }

  #[test]
  fn istft_named_window_is_periodic_no_trailing_zero() {
    // Regression on the `window_fn(win_length + 1)[:-1]` periodic
    // construction: the synthesis window must have its trailing sample
    // dropped (so it is NOT the symmetric window with a zero at the end).
    // We can observe this only indirectly via the reconstruction, so instead
    // assert the construction directly here for win_length=8:
    // periodic hann(8) = hann(9)[:-1] = [0, .1464.., .5, .8535.., 1, .8535..,
    // .5, .1464..] — note the LAST sample is 0.1464.., not 0.
    let full = hann_window(9).unwrap();
    let periodic = ops::indexing::slice(&full, &[0], &[8], &[1]).unwrap();
    let v = to_vec(&periodic);
    assert_eq!(v.len(), 8);
    assert!(
      v[0].abs() < WIN_TOL,
      "periodic[0] should be 0, got {}",
      v[0]
    );
    assert!(
      (v[7] - 0.146_447).abs() < 1e-4,
      "periodic[7] should be ~0.1464 (NOT 0), got {}",
      v[7]
    );
  }

  #[test]
  fn istft_rejects_bad_shapes_and_params() {
    // 1-D input (must be 2-D).
    let one_d = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
    assert!(matches!(
      istft(&one_d, None, None, Window::Named("hann"), true, None, false),
      Err(Error::Backend { .. })
    ));

    // Valid spec for the remaining param checks.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, None).unwrap();
    let w = hann_window(8).unwrap();
    // hop_length == 0.
    assert!(matches!(
      istft(
        &spec,
        Some(0),
        Some(8),
        Window::Array(&w),
        true,
        None,
        false
      ),
      Err(Error::Backend { .. })
    ));
    // win_length > n_fft (n_fft = 8 here).
    assert!(matches!(
      istft(
        &spec,
        Some(4),
        Some(16),
        Window::Array(&w),
        true,
        None,
        false
      ),
      Err(Error::Backend { .. })
    ));
    // length larger than the OLA length (t = 24).
    assert!(matches!(
      istft(
        &spec,
        Some(4),
        Some(8),
        Window::Array(&w),
        true,
        Some(1000),
        true
      ),
      Err(Error::Backend { .. })
    ));
  }

  /// A 19-sample fixed test signal for the non-hop-aligned round-trips.
  fn signal_19() -> [f32; 19] {
    [
      0.1, 0.5, -0.3, 0.8, -0.2, 0.6, 0.0, -0.7, 0.4, 0.9, -0.5, 0.2, 0.3, -0.1, 0.7, -0.4, 0.55,
      0.66, -0.77,
    ]
  }

  #[test]
  fn istft_roundtrip_non_hop_aligned_length_asserts_values() {
    // Codex `length` finding (the un-masked bug): when the original length is
    // NOT a multiple of `hop`, the centered reconstruction must still recover
    // every original sample VALUE — but only if the center pad is removed
    // BEFORE `length` (returning `reconstructed[pad .. pad + length]`,
    // `pad = n_fft/2`). With the pre-fix `reconstructed[0 .. length]` the head
    // started inside the reflected prefix and the tail was dropped.
    //
    // n_fft=8, hop=4, win_length=8. For L=17: padded=25, num_frames=5,
    // t=(5-1)*4+8=24, centered region [4 .. 20] = 16 samples — so `length=None`
    // would silently SHORTEN a 17-sample input to 16. `length=Some(17)`
    // recovers all 17. Cross-checked against an f64 numpy mirror of stft/istft
    // (max error 2.2e-16); we assert 1e-5 for the f32 backend.
    for &len in &[17usize, 19usize] {
      let full = signal_19();
      let buf = &full[..len];
      let x = Array::from_slice::<f32>(buf, &[len as i32]).unwrap();
      let spec = stft(&x, 8, 4, None).unwrap();
      // num_frames == 5 for both 17 and 19 (padded 25 / 27, (25-8)/4 ==
      // (27-8)/4 == 4 → 5 frames); n_freqs == 5.
      assert_eq!(spec.shape(), vec![5, 5]);
      let w = hann_window(8).unwrap();
      let rec = istft(
        &spec,
        Some(4),
        Some(8),
        Window::Array(&w),
        true,      // center → drop the n_fft/2 reflect-pad first
        Some(len), // length → keep exactly `len` real samples
        true,      // normalized (Σw²)
      )
      .unwrap();
      let r = to_vec(&rec);
      assert_eq!(
        r.len(),
        len,
        "non-hop-aligned length {len}: wrong output length"
      );
      for (i, (g, e)) in r.iter().zip(buf.iter()).enumerate() {
        assert!(
          (g - e).abs() < 1e-5,
          "non-hop-aligned[{len}] reconstruction[{i}]: got {g}, want {e} (diff {})",
          (g - e).abs()
        );
      }
    }
  }

  #[test]
  fn istft_roundtrip_win_length_lt_n_fft_asserts_values() {
    // Codex `win_length < n_fft` finding: the FAITHFUL inverse overlap-adds
    // full `n_fft`-wide frames against the synthesis window padded to `n_fft`
    // exactly as `stft` pads its analysis window, and trims by `n_fft/2`. We
    // pass the SAME base window `stft` uses (symmetric Hann of `win_length`)
    // via Window::Array + normalized=true, so each frame contributes `w²·x`
    // and the divisor is `Σ w²` → exact `x` over the COLA-covered span.
    //
    // n_fft=16, win_length=8, hop=4 on a 16-sample signal: padded=32,
    // num_frames=5, n_freqs=9, t=(5-1)*4+16=32, centered region [8 .. 24] = 16
    // samples. The analysis window is only nonzero in its first 8 of 16
    // samples, so the LAST centered sample (index 15) has window-sum 0 and is
    // left un-normalized (intrinsic to the zero-padded window, NOT a defect);
    // every covered sample [0 .. 15) reconstructs to the original. Verified
    // against an f64 numpy mirror (covered-region max error 1.1e-16). The
    // pre-fix code (slice frames to win_length, trim win_length/2) produced a
    // SHIFTED/corrupt signal here (error ~0.4 across the board).
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 16, 4, Some(8)).unwrap();
    assert_eq!(spec.shape(), vec![5, 9]); // (num_frames, n_fft/2+1)
    let w = hann_window(8).unwrap(); // same base window stft uses (win_length=8)
    let rec = istft(
      &spec,
      Some(4), // hop_length (matches stft)
      Some(8), // win_length < n_fft
      Window::Array(&w),
      true,     // center → trim n_fft/2 = 8
      Some(16), // length → keep all 16 (covered region is [0..15))
      true,     // normalized (Σw²)
    )
    .unwrap();
    let r = to_vec(&rec);
    assert_eq!(r.len(), 16, "win<n_fft round-trip length mismatch");
    // Assert VALUES over the COLA-covered span [0 .. 15). The last sample is
    // outside the window's coverage (window-sum 0) and is intentionally not
    // asserted here.
    for (i, (g, e)) in r.iter().zip(buf.iter()).take(15).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "win<n_fft reconstruction[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  #[test]
  fn istft_roundtrip_win_length_lt_n_fft_full_coverage_exact() {
    // Companion to the above: with a wider window (win_length=12 < n_fft=16,
    // hop=4) the zero-padded synthesis window DOES satisfy COLA across the
    // whole centered region, so ALL 16 samples reconstruct exactly. This pins
    // the faithful `win_length < n_fft` inverse end-to-end with no
    // boundary-coverage caveat (f64 numpy mirror: max error 2.2e-16).
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 16, 4, Some(12)).unwrap();
    assert_eq!(spec.shape(), vec![5, 9]);
    let w = hann_window(12).unwrap();
    let rec = istft(
      &spec,
      Some(4),
      Some(12),
      Window::Array(&w),
      true,
      Some(16),
      true,
    )
    .unwrap();
    let r = to_vec(&rec);
    assert_eq!(r.len(), 16);
    for (i, (g, e)) in r.iter().zip(buf.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "win12<n_fft16 reconstruction[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  #[test]
  fn istft_rejects_pathological_scatter_work_before_alloc() {
    // Codex OOM finding: the real scatter/update workload is
    // `num_frames * n_fft`, which can dwarf the OLA *output* length `t` for
    // small hops. The `t <= MAX_DECODED_SAMPLES` cap does NOT catch this; the
    // dedicated MAX_OLA_WORK guard must reject it BEFORE any
    // broadcast/flatten/`try_reserve` (and before the irfft).
    //
    // We pick a shape where `t` is small but the work explodes, using a LAZY
    // mlx spectrum (`zeros(...).astype(Complex64)`) so nothing is materialized
    // — exactly the "shaped/lazy input" the finding describes. The cap fires
    // off `x.shape()` alone, so the (logically ~600 MiB) spectrum is never
    // allocated, the synthesis-window pads stay lazy, and the irfft never
    // runs.
    //
    // num_frames=4, n_freqs=9 Mi+1 → n_fft=(n_freqs-1)*2=18 Mi.
    //   work = num_frames * n_fft = 4 * 18 Mi = 72 Mi  > MAX_OLA_WORK (64 Mi) ✓
    //   t    = (4-1)*hop + n_fft  = 6 + 18 Mi ≈ 18 Mi  < MAX_DECODED  (64 Mi)
    // so ONLY the work cap can reject this — proving it is the work cap, not
    // the output-length cap, doing the job. `win_length=Some(8)` +
    // Window::Array keeps window construction trivial (no huge Vec).
    let n_freqs: i32 = 9 * 1024 * 1024 + 1;
    let num_frames: i32 = 4;
    let spec = Array::zeros::<f32>(&[num_frames, n_freqs])
      .unwrap()
      .astype(crate::Dtype::Complex64)
      .unwrap();
    let w = hann_window(8).unwrap();
    let res = istft(
      &spec,
      Some(2), // small hop → t stays under the decoded cap
      Some(8), // tiny win_length → cheap window, padded lazily to n_fft
      Window::Array(&w),
      true,
      None,
      false,
    );
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "pathological num_frames*n_fft must be rejected by the MAX_OLA_WORK cap"
    );
  }
}
