//! Fully-fallible, PIL-matching RGBA8 image resize (own implementation).
//!
//! This module replaces the third-party `fast_image_resize` crate that
//! [`crate::vlm::image::resize`] previously delegated to. The motivation
//! is allocation safety, not performance parity: `resize`'s target
//! dimensions flow from an UNTRUSTED loaded `preprocessor_config.json`
//! (see [`crate::vlm::load`]), and `fast_image_resize` allocated internal
//! scratch (coefficient tables, per-row work buffers) *infallibly* inside
//! the crate — a hostile-but-under-cap target could `abort()` the process
//! despite our `Result` signature. Owning the whole resize lets EVERY
//! allocation route through `try_reserve_exact`, so `resize` returning
//! `Ok` guarantees no abort path for any (untrusted) target size.
//!
//! ## Correctness reference — PIL `Image.resize`
//! mlx-vlm preprocessing expects **PIL `Image.resize`** semantics (the
//! swift `MediaProcessing.resampleBicubic` mirrors PIL). The convolution
//! filters here reproduce PIL's `src/libImaging/Resample.c` *exactly*,
//! including its fixed-point integer accumulation, so the output is
//! **byte-for-byte identical to PIL** (verified against Pillow 12.2 over
//! bilinear/bicubic/lanczos, upscale + downscale, RGBA — see
//! `tests/vlm_image.rs`). No ±1 LSB tolerance is required for the scalar
//! path; it is bit-exact with PIL.
//!
//! ### Algorithm (matches `Resample.c`)
//! Separable two-pass convolution: a horizontal 1-D pass that emits an
//! 8-bit clamped intermediate image, then a vertical 1-D pass over that
//! intermediate. For each output coordinate the value is a weighted sum
//! of input pixels within the filter's support window, weights from the
//! filter kernel normalized to sum to 1.
//!
//! ### Coordinate mapping + antialiasing (matches `precompute_coeffs`)
//! For output index `xx` along an axis resampled from `in_size` to
//! `out_size`:
//! - `scale = in_size / out_size`
//! - `center = (xx + 0.5) * scale`
//! - `filterscale = max(scale, 1.0)` — the **antialiasing filter-stretch**:
//!   when downscaling (`scale > 1`), the filter support widens by the
//!   scale factor so the kernel averages over the shrinking footprint.
//! - `support = filter_support * filterscale`
//! - window `[floor(center - support + 0.5), floor(center + support + 0.5))`
//!   clamped to `[0, in_size)`
//! - weight for input `x` in the window:
//!   `filter((x - center + 0.5) / filterscale)`, then all weights in the
//!   window normalized so they sum to 1.0.
//!
//! ### Fixed-point accumulation (matches `Resample.c` `clip8`)
//! PIL normalizes the f64 weights to fixed point with
//! `PRECISION_BITS = 22`: `coef_i = round(coef * (1 << 22))` (an `i32`).
//! The per-output accumulator is an `i32` seeded with the rounding bias
//! `1 << (PRECISION_BITS - 1)`, accumulates `pixel * coef_i`, then is
//! finished with an **arithmetic** `>> PRECISION_BITS` (sign-extending,
//! matching C's signed shift) and clamped to `[0, 255]`. The `i32`
//! accumulator does not overflow: the worst-case partial sum for these
//! kernels is `≈ 255 * 1.2 * (1 << 22) ≈ 1.28e9 < i32::MAX ≈ 2.15e9`
//! (the `sum(|coef|)` over each window is `< 1.2` for Keys-cubic a=-0.5
//! and Lanczos a=3; the filterscale spreads coefficients but shrinks each
//! so the bound holds at any scale).
//!
//! ### Nearest
//! PIL's `NEAREST` resize maps output index `o` to input
//! `min(floor((o + 0.5) * in_size / out_size), in_size - 1)` (verified
//! against Pillow). It is a pure pixel gather — no convolution, no
//! coefficient table.
//!
//! ## SIMD
//! The hot loop is the inner per-output-pixel weighted sum over the
//! support window, per channel. RGBA8 is `[u8; 4]` per pixel, so the NEON
//! kernel vectorizes **across the 4 channels**: widen the 4 source bytes
//! to `int32x4`, fused-multiply-accumulate by the (broadcast) `i32`
//! coefficient into an `int32x4` accumulator, then narrow back to 4 `u8`
//! with the same arithmetic shift + clamp. This produces output
//! bit-identical to the scalar path (same `i32` math, same rounding).
//! The coefficient precomputation (cold, once per resize) stays scalar.
//!
//! Per the project SIMD conventions: NEON is gated on
//! `#[cfg(target_arch = "aarch64")]` + a runtime
//! `is_aarch64_feature_detected!("neon")` check, the scalar fallback is
//! ALWAYS compiled, the `#[target_feature(enable = "neon")] unsafe fn`
//! kernels carry numbered `# Safety` clauses, slice-length preconditions
//! are `assert!`ed unconditionally, and the `--cfg mlxrs_force_scalar`
//! escape forces the scalar path even on aarch64. There is NO cargo
//! feature: the dispatch is always-on. (This is self-contained in `vlm`;
//! it can be refactored into a shared `mlxrs::simd` module later.)

use crate::error::{Error, Result, try_with_capacity};

/// Interpolation filter for [`resize_rgba8`], mirroring PIL's resampling
/// filters. The variants line up 1:1 with
/// [`crate::vlm::image::ResizeFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Filter {
  /// Nearest-neighbor pixel gather (no smoothing). PIL `Image.NEAREST`.
  Nearest,
  /// Triangle / linear kernel, support `1.0`. PIL `Image.BILINEAR`.
  Bilinear,
  /// Keys cubic with `a = -0.5`, support `2.0`. PIL `Image.BICUBIC`.
  Bicubic,
  /// Sinc-windowed sinc with `a = 3`, support `3.0`. PIL `Image.LANCZOS`.
  Lanczos3,
}

/// PIL fixed-point precision: `coef_int = round(coef * (1 << 22))`, and
/// the accumulator is finished with `>> 22`. Matches `Resample.c`'s
/// `#define PRECISION_BITS (32 - 8 - 2)`.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Rounding bias added to the fixed-point accumulator before the final
/// shift (`1 << (PRECISION_BITS - 1)`), matching `Resample.c`.
const ROUND_BIAS: i32 = 1 << (PRECISION_BITS - 1);

/// RGBA8 has 4 channels (the only pixel layout this module handles — the
/// caller materializes every source variant to RGBA8 first).
const CHANNELS: usize = 4;

/// Continuous filter support radius (the half-width of the kernel before
/// the antialiasing filterscale stretch).
fn filter_support(f: Filter) -> f64 {
  match f {
    // Nearest has no continuous kernel; never queried (handled separately).
    Filter::Nearest => 0.0,
    Filter::Bilinear => 1.0,
    Filter::Bicubic => 2.0,
    Filter::Lanczos3 => 3.0,
  }
}

/// Evaluate the continuous filter kernel at `x` (already divided by the
/// filterscale by the caller). Each matches PIL's `Resample.c`:
/// - Bilinear: triangle `1 - |x|` on `[-1, 1]`.
/// - Bicubic: Keys cubic with `a = -0.5`.
/// - Lanczos3: `sinc(x) * sinc(x / 3)` on `[-3, 3]`.
fn filter_eval(f: Filter, x: f64) -> f64 {
  match f {
    Filter::Nearest => 0.0,
    Filter::Bilinear => {
      let x = x.abs();
      if x < 1.0 { 1.0 - x } else { 0.0 }
    }
    Filter::Bicubic => {
      // PIL Keys cubic, a = -0.5.
      const A: f64 = -0.5;
      let x = x.abs();
      if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
      } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
      } else {
        0.0
      }
    }
    Filter::Lanczos3 => {
      let x = x.abs();
      if x < 3.0 {
        sinc(x) * sinc(x / 3.0)
      } else {
        0.0
      }
    }
  }
}

/// Normalized sinc, `sin(pi x) / (pi x)`, with `sinc(0) = 1` — matching
/// PIL's `sinc_filter`.
fn sinc(x: f64) -> f64 {
  if x == 0.0 {
    1.0
  } else {
    let px = x * std::f64::consts::PI;
    px.sin() / px
  }
}

/// Precomputed per-output-index convolution coefficients for one axis.
///
/// `bounds[o] = (xmin, n)` gives the input window start and length for
/// output index `o`; `weights[o * ksize .. o * ksize + n]` are the
/// fixed-point `i32` taps for that output (the remaining `ksize - n`
/// slots in the row are zero-padded so every row has a uniform stride —
/// this keeps the convolution inner loop branch-free on row stride).
///
/// All three backing `Vec`s are reserved via `try_reserve_exact`; this
/// type is the "coefficient table" `fast_image_resize` allocated
/// infallibly.
struct Coeffs {
  /// `(xmin, n)` per output index.
  bounds: Vec<(usize, usize)>,
  /// Fixed-point taps, row-major with stride `ksize`.
  weights: Vec<i32>,
  /// Per-output row stride (`max` window length across outputs).
  ksize: usize,
}

/// Precompute the convolution coefficients for resampling one axis from
/// `in_size` to `out_size` with `filter` (PIL `precompute_coeffs` +
/// `normalize_coeffs_8bpc`).
///
/// Every buffer is `try_reserve_exact`-backed; an allocator refusal
/// surfaces as [`Error::OutOfMemory`]. A degenerate `in_size`/`out_size`
/// (zero) or a `ksize` overflow surfaces as [`Error::ShapeMismatch`].
fn precompute_coeffs(in_size: usize, out_size: usize, filter: Filter) -> Result<Coeffs> {
  // Caller guarantees non-zero, but guard defensively: a zero `out_size`
  // would divide by zero in `scale`, a zero `in_size` makes the window
  // empty.
  if in_size == 0 || out_size == 0 {
    return Err(Error::ShapeMismatch {
      message: format!("precompute_coeffs: in_size={in_size} out_size={out_size} must be non-zero"),
    });
  }
  let scale = in_size as f64 / out_size as f64;
  let filterscale = if scale < 1.0 { 1.0 } else { scale };
  let support = filter_support(filter) * filterscale;
  // `ksize` is the max number of taps any output index can reference:
  // `ceil(support) * 2 + 1`, exactly PIL's `ksize = (int)ceil(support) *
  // 2 + 1`. Bounded by `in_size` (a window can never exceed the input).
  let ksize_unclamped = (support.ceil() as usize)
    .checked_mul(2)
    .and_then(|v| v.checked_add(1))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("precompute_coeffs: ksize overflows for support={support}"),
    })?;
  let ksize = ksize_unclamped.min(in_size.max(1));

  let mut bounds: Vec<(usize, usize)> = try_with_capacity(out_size)?;
  // `out_size * ksize` weights, fallibly. `checked_mul` so a hostile
  // product (already bounded by the caller's MAX_DECODED cap, but be
  // explicit) routes to a recoverable error rather than a wrap.
  let weight_len = out_size
    .checked_mul(ksize)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("precompute_coeffs: out_size*ksize overflows for {out_size}*{ksize}"),
    })?;
  let mut weights: Vec<i32> = try_with_capacity(weight_len)?;
  weights.resize(weight_len, 0i32);

  // Scratch for one row of f64 weights before fixed-point conversion.
  // Bounded by `ksize` (small), reserved fallibly.
  let mut row: Vec<f64> = try_with_capacity(ksize)?;

  let inv_filterscale = 1.0 / filterscale;
  for xx in 0..out_size {
    let center = (xx as f64 + 0.5) * scale;
    // Window `[xmin, xmax)` clamped to `[0, in_size)`. PIL adds 0.5 and
    // truncates toward zero; `center - support` is >= 0 here only after
    // the clamp, and the `+ 0.5` then `as usize`/`as i64` truncation
    // matches C's `(int)`.
    let xmin = {
      let v = (center - support + 0.5).floor();
      if v < 0.0 { 0 } else { v as usize }
    };
    let xmax = {
      let v = (center + support + 0.5).floor();
      let v = if v < 0.0 { 0usize } else { v as usize };
      v.min(in_size)
    };
    let n = xmax.saturating_sub(xmin);
    // Accumulate raw weights, then normalize to sum 1.0 (PIL divides
    // each tap by the window sum).
    row.clear();
    let mut wsum = 0.0f64;
    for i in 0..n {
      let w = filter_eval(
        filter,
        (xmin as f64 + i as f64 - center + 0.5) * inv_filterscale,
      );
      row.push(w);
      wsum += w;
    }
    let base = xx * ksize;
    if wsum != 0.0 {
      let inv = 1.0 / wsum;
      for (i, &w) in row.iter().enumerate() {
        // Fixed-point: round(coef * (1 << PRECISION_BITS)).
        let scaled = (w * inv) * f64::from(1i32 << PRECISION_BITS);
        weights[base + i] = scaled.round() as i32;
      }
    }
    // n is bounded by ksize by construction (window <= ceil(support)*2+1
    // and clamped to in_size). Assert to make the convolution's slice
    // access provably in-bounds.
    debug_assert!(
      n <= ksize,
      "precompute_coeffs: window n={n} exceeds ksize={ksize}"
    );
    bounds.push((xmin, n));
  }
  Ok(Coeffs {
    bounds,
    weights,
    ksize,
  })
}

/// Clamp a finished fixed-point accumulator to `u8` exactly as PIL's
/// `clip8`: arithmetic `>> PRECISION_BITS` (sign-extending) then clamp to
/// `[0, 255]`.
#[inline]
fn clip8(acc: i32) -> u8 {
  // Rust `>>` on `i32` is arithmetic (sign-preserving), matching C's
  // signed right shift used by `clip8`.
  let v = acc >> PRECISION_BITS;
  if v < 0 {
    0
  } else if v > 255 {
    255
  } else {
    v as u8
  }
}

/// Resize an RGBA8 image from `(src_w, src_h)` to `(dst_w, dst_h)` using
/// `filter`. `src` MUST be exactly `src_w * src_h * 4` bytes; the returned
/// `Vec<u8>` is exactly `dst_w * dst_h * 4` bytes (row-major RGBA8).
///
/// EVERY buffer (coefficient tables for both axes, the horizontal-pass
/// intermediate, the output) is `try_reserve_exact`-backed; an allocator
/// refusal surfaces as [`Error::OutOfMemory`], never a process abort.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if any dimension is `0`, if a byte/element
///   product overflows `usize`, or if `src.len() != src_w * src_h * 4`.
/// - [`Error::OutOfMemory`] if any `try_reserve_exact` fails.
///
/// # Panics
/// Does not panic on valid input: the only `assert!`s are slice-length
/// preconditions inside the SIMD/scalar kernels, which the dimension math
/// in this function makes structurally true.
pub(crate) fn resize_rgba8(
  src: &[u8],
  src_w: usize,
  src_h: usize,
  dst_w: usize,
  dst_h: usize,
  filter: Filter,
) -> Result<Vec<u8>> {
  if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "resize_rgba8: dimensions must be non-zero, got src {src_w}x{src_h} dst {dst_w}x{dst_h}"
      ),
    });
  }
  let src_len = src_w
    .checked_mul(src_h)
    .and_then(|v| v.checked_mul(CHANNELS))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize_rgba8: src_w*src_h*4 overflows usize for {src_w}x{src_h}"),
    })?;
  if src.len() != src_len {
    return Err(Error::ShapeMismatch {
      message: format!(
        "resize_rgba8: src buffer is {} bytes, expected src_w*src_h*4={src_len} for {src_w}x{src_h}",
        src.len()
      ),
    });
  }
  let dst_len = dst_w
    .checked_mul(dst_h)
    .and_then(|v| v.checked_mul(CHANNELS))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize_rgba8: dst_w*dst_h*4 overflows usize for {dst_w}x{dst_h}"),
    })?;

  if filter == Filter::Nearest {
    return resize_nearest(src, src_w, src_h, dst_w, dst_h, dst_len);
  }

  // --- Separable convolution ---
  // Horizontal pass: (src_h rows) x (dst_w cols) intermediate, RGBA8.
  // Vertical pass: (dst_h rows) x (dst_w cols) output.
  let hcoeffs = precompute_coeffs(src_w, dst_w, filter)?;
  let vcoeffs = precompute_coeffs(src_h, dst_h, filter)?;

  // Intermediate buffer: src_h * dst_w * 4 bytes, fallible. (PIL emits an
  // 8-bit clamped image between the two passes; the vertical pass reads
  // it back.)
  let inter_len = src_h
    .checked_mul(dst_w)
    .and_then(|v| v.checked_mul(CHANNELS))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize_rgba8: src_h*dst_w*4 overflows usize for {src_h}x{dst_w}"),
    })?;
  let mut inter: Vec<u8> = try_with_capacity(inter_len)?;
  inter.resize(inter_len, 0u8);

  // Output buffer, fallible.
  let mut dst: Vec<u8> = try_with_capacity(dst_len)?;
  dst.resize(dst_len, 0u8);

  // Horizontal pass: for each src row, convolve along x into `inter`.
  convolve_axis(src, src_w, src_h, &mut inter, dst_w, &hcoeffs);
  // Vertical pass: convolve `inter` along y into `dst`. We transpose the
  // access by treating columns: for each output row `oy`, gather input
  // rows `[ymin, ymin+n)` from `inter`. To reuse `convolve_axis` (which
  // convolves along the contiguous x-axis), the vertical pass is a
  // separate routine because its taps stride by a full row.
  convolve_vertical(&inter, dst_w, src_h, &mut dst, dst_h, &vcoeffs);

  Ok(dst)
}

/// Nearest-neighbor resize (pure pixel gather, PIL `Image.NEAREST`).
/// Output index `o` maps to input `min(floor((o+0.5)*in/out), in-1)`.
fn resize_nearest(
  src: &[u8],
  src_w: usize,
  src_h: usize,
  dst_w: usize,
  dst_h: usize,
  dst_len: usize,
) -> Result<Vec<u8>> {
  // Precompute per-output-column source x indices (fallible, small).
  let mut xmap: Vec<usize> = try_with_capacity(dst_w)?;
  for ox in 0..dst_w {
    let sx = ((ox as f64 + 0.5) * src_w as f64 / dst_w as f64).floor() as usize;
    xmap.push(sx.min(src_w - 1));
  }
  let mut dst: Vec<u8> = try_with_capacity(dst_len)?;
  dst.resize(dst_len, 0u8);
  for oy in 0..dst_h {
    let sy = (((oy as f64 + 0.5) * src_h as f64 / dst_h as f64).floor() as usize).min(src_h - 1);
    let src_row = &src[sy * src_w * CHANNELS..(sy + 1) * src_w * CHANNELS];
    let dst_row = &mut dst[oy * dst_w * CHANNELS..(oy + 1) * dst_w * CHANNELS];
    for ox in 0..dst_w {
      let sx = xmap[ox];
      dst_row[ox * CHANNELS..ox * CHANNELS + CHANNELS]
        .copy_from_slice(&src_row[sx * CHANNELS..sx * CHANNELS + CHANNELS]);
    }
  }
  Ok(dst)
}

/// Horizontal convolution: for each of `rows` source rows, produce
/// `out_w` output pixels into `out` (RGBA8, `rows * out_w * 4` bytes).
/// Dispatches to the NEON kernel on aarch64 (unless `mlxrs_force_scalar`),
/// else the scalar kernel.
fn convolve_axis(
  src: &[u8],
  src_w: usize,
  rows: usize,
  out: &mut [u8],
  out_w: usize,
  coeffs: &Coeffs,
) {
  // Slice-length preconditions (unconditional assert per SIMD conventions):
  // both kernels rely on these to keep their indexing in-bounds.
  assert_eq!(src.len(), src_w * rows * CHANNELS, "convolve_axis: src len");
  assert_eq!(out.len(), out_w * rows * CHANNELS, "convolve_axis: out len");
  assert_eq!(coeffs.bounds.len(), out_w, "convolve_axis: bounds len");

  #[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
  {
    if std::arch::is_aarch64_feature_detected!("neon") {
      // SAFETY: the `neon` target feature is confirmed available by the
      // runtime `is_aarch64_feature_detected!` check immediately above;
      // see `convolve_axis_neon`'s `# Safety` for the full contract.
      unsafe {
        convolve_axis_neon(src, src_w, rows, out, out_w, coeffs);
      }
      return;
    }
  }
  convolve_axis_scalar(src, src_w, rows, out, out_w, coeffs);
}

/// Vertical convolution: read the `src_h x out_w` intermediate `inter`
/// and produce `out_h` output rows into `out` (RGBA8). Taps stride by a
/// full intermediate row.
fn convolve_vertical(
  inter: &[u8],
  out_w: usize,
  src_h: usize,
  out: &mut [u8],
  out_h: usize,
  coeffs: &Coeffs,
) {
  assert_eq!(
    inter.len(),
    out_w * src_h * CHANNELS,
    "convolve_vertical: inter len"
  );
  assert_eq!(
    out.len(),
    out_w * out_h * CHANNELS,
    "convolve_vertical: out len"
  );
  assert_eq!(coeffs.bounds.len(), out_h, "convolve_vertical: bounds len");

  #[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
  {
    if std::arch::is_aarch64_feature_detected!("neon") {
      // SAFETY: `neon` confirmed by the runtime check above; see
      // `convolve_vertical_neon`'s `# Safety`.
      unsafe {
        convolve_vertical_neon(inter, out_w, src_h, out, out_h, coeffs);
      }
      return;
    }
  }
  convolve_vertical_scalar(inter, out_w, src_h, out, out_h, coeffs);
}

/// Scalar horizontal convolution (always compiled). Bit-exact with PIL.
fn convolve_axis_scalar(
  src: &[u8],
  src_w: usize,
  rows: usize,
  out: &mut [u8],
  out_w: usize,
  coeffs: &Coeffs,
) {
  let ksize = coeffs.ksize;
  for y in 0..rows {
    let src_row = &src[y * src_w * CHANNELS..(y + 1) * src_w * CHANNELS];
    let out_row = &mut out[y * out_w * CHANNELS..(y + 1) * out_w * CHANNELS];
    for ox in 0..out_w {
      let (xmin, n) = coeffs.bounds[ox];
      let taps = &coeffs.weights[ox * ksize..ox * ksize + n];
      let mut acc = [ROUND_BIAS; CHANNELS];
      for (i, &w) in taps.iter().enumerate() {
        let px = &src_row[(xmin + i) * CHANNELS..(xmin + i) * CHANNELS + CHANNELS];
        acc[0] += i32::from(px[0]) * w;
        acc[1] += i32::from(px[1]) * w;
        acc[2] += i32::from(px[2]) * w;
        acc[3] += i32::from(px[3]) * w;
      }
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      o[0] = clip8(acc[0]);
      o[1] = clip8(acc[1]);
      o[2] = clip8(acc[2]);
      o[3] = clip8(acc[3]);
    }
  }
}

/// Scalar vertical convolution (always compiled). Bit-exact with PIL.
fn convolve_vertical_scalar(
  inter: &[u8],
  out_w: usize,
  _src_h: usize,
  out: &mut [u8],
  out_h: usize,
  coeffs: &Coeffs,
) {
  let ksize = coeffs.ksize;
  let row_stride = out_w * CHANNELS;
  for oy in 0..out_h {
    let (ymin, n) = coeffs.bounds[oy];
    let taps = &coeffs.weights[oy * ksize..oy * ksize + n];
    let out_row = &mut out[oy * row_stride..(oy + 1) * row_stride];
    for ox in 0..out_w {
      let mut acc = [ROUND_BIAS; CHANNELS];
      for (i, &w) in taps.iter().enumerate() {
        let base = (ymin + i) * row_stride + ox * CHANNELS;
        let px = &inter[base..base + CHANNELS];
        acc[0] += i32::from(px[0]) * w;
        acc[1] += i32::from(px[1]) * w;
        acc[2] += i32::from(px[2]) * w;
        acc[3] += i32::from(px[3]) * w;
      }
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      o[0] = clip8(acc[0]);
      o[1] = clip8(acc[1]);
      o[2] = clip8(acc[2]);
      o[3] = clip8(acc[3]);
    }
  }
}

/// NEON horizontal convolution. Vectorizes the per-output weighted sum
/// across the 4 RGBA channels: widen the 4 source bytes to `int32x4`,
/// multiply-accumulate by the broadcast `i32` coefficient, then narrow +
/// shift + clamp back to 4 `u8`. Output is bit-identical to
/// [`convolve_axis_scalar`] (identical `i32` arithmetic + rounding).
///
/// # Safety
/// 1. The `neon` target feature must be available at runtime. The sole
///    caller ([`convolve_axis`]) gates this on
///    `is_aarch64_feature_detected!("neon")`, so the `vld*`/`vmlaq`/etc.
///    intrinsics are legal on the executing CPU.
/// 2. `src.len() == src_w * rows * 4`, `out.len() == out_w * rows * 4`,
///    and `coeffs.bounds.len() == out_w` — all asserted unconditionally
///    by the caller before dispatch. Combined with the
///    [`precompute_coeffs`] invariant `xmin + n <= src_w` (window clamped
///    to the input), every byte slice accessed below is in-bounds.
/// 3. All loads/stores are 4-byte (one RGBA8 pixel) and operate on the
///    `&[u8]`/`&mut [u8]` slices directly (no raw pointer aliasing beyond
///    the borrow the references already grant).
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn convolve_axis_neon(
  src: &[u8],
  src_w: usize,
  rows: usize,
  out: &mut [u8],
  out_w: usize,
  coeffs: &Coeffs,
) {
  use std::arch::aarch64::*;
  let ksize = coeffs.ksize;
  for y in 0..rows {
    let src_row = &src[y * src_w * CHANNELS..(y + 1) * src_w * CHANNELS];
    let out_row = &mut out[y * out_w * CHANNELS..(y + 1) * out_w * CHANNELS];
    for ox in 0..out_w {
      let (xmin, n) = coeffs.bounds[ox];
      let taps = &coeffs.weights[ox * ksize..ox * ksize + n];
      // Seed all four lanes with the rounding bias. Value-only NEON
      // intrinsics need no `unsafe` block inside a `#[target_feature]`
      // fn — the feature gate discharges their safety; only the pointer
      // load/store below carry an `unsafe {}` (with a SAFETY note).
      let mut acc = vdupq_n_s32(ROUND_BIAS);
      for (i, &w) in taps.iter().enumerate() {
        let off = (xmin + i) * CHANNELS;
        // `off + 4 <= src_row.len()` by the window invariant
        // (`xmin + n <= src_w`, asserted via Safety clause 2).
        let px4 = [
          src_row[off],
          src_row[off + 1],
          src_row[off + 2],
          src_row[off + 3],
        ];
        // SAFETY: clauses 1+3 — `neon` confirmed by the dispatch gate;
        // `neon_load_rgba` zero-extends 4 RGBA bytes into a `uint8x8_t`
        // and only reads its own 8-byte stack array.
        let v8 = unsafe { neon_load_rgba(px4) };
        let v16 = vmovl_u8(v8); // u8x8 -> u16x8
        let v16lo = vget_low_u16(v16); // first 4 u16 (R,G,B,A)
        let v32 = vreinterpretq_s32_u32(vmovl_u16(v16lo)); // u16x4 -> s32x4
        let wv = vdupq_n_s32(w);
        acc = vmlaq_s32(acc, v32, wv);
      }
      // Arithmetic shift right by PRECISION_BITS (matches scalar `>>`),
      // then narrow with unsigned saturation to u8 (clamps to [0,255],
      // matching `clip8`): `vqmovun_s32` maps negatives to 0, the
      // subsequent `vqmovn_u16` saturates the > 255 case.
      let shifted = vshrq_n_s32::<{ PRECISION_BITS as i32 }>(acc);
      let u16x4 = vqmovun_s32(shifted); // s32x4 -> u16x4 (sat, >=0)
      let u16x8 = vcombine_u16(u16x4, vdup_n_u16(0));
      let u8x8 = vqmovn_u16(u16x8); // u16x8 -> u8x8 (sat to 255)
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      // SAFETY: clauses 1+3 — `neon` confirmed by the dispatch gate;
      // `neon_store_rgba` writes only its own 8-byte stack array and `o`
      // is exactly `CHANNELS` bytes (asserted inside the helper).
      unsafe { neon_store_rgba(u8x8, o) };
    }
  }
}

/// Load 4 RGBA bytes into the low half of a `uint8x8_t` (high 4 lanes
/// zero). Isolates the only pointer-based NEON `unsafe` in the kernels.
///
/// # Safety
/// 1. `neon` available at runtime (the kernels are reached only after the
///    dispatch gate's `is_aarch64_feature_detected!("neon")`).
/// 2. Reads exactly 8 bytes from an 8-byte stack array — fully in-bounds.
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn neon_load_rgba(px4: [u8; CHANNELS]) -> std::arch::aarch64::uint8x8_t {
  use std::arch::aarch64::*;
  // Widen to 8 bytes (low 4 = pixel, high 4 = 0) so the single 8-byte
  // `vld1_u8` reads only initialized stack memory.
  let buf = [px4[0], px4[1], px4[2], px4[3], 0, 0, 0, 0];
  // SAFETY: clauses 1+2 — `vld1_u8` reads 8 bytes from `buf` (`[u8; 8]`),
  // all initialized and in-bounds; `neon` confirmed by the dispatch gate.
  unsafe { vld1_u8(buf.as_ptr()) }
}

/// Store the low 4 lanes of a `uint8x8_t` into a 4-byte RGBA output slice.
///
/// # Safety
/// 1. `neon` available at runtime (see [`neon_load_rgba`]).
/// 2. `out.len() == 4` (one RGBA pixel) — the kernels slice exactly
///    `CHANNELS` bytes.
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn neon_store_rgba(v: std::arch::aarch64::uint8x8_t, out: &mut [u8]) {
  use std::arch::aarch64::*;
  assert_eq!(
    out.len(),
    CHANNELS,
    "neon_store_rgba: out must be one RGBA pixel"
  );
  let mut tmp = [0u8; 8];
  // SAFETY: clauses 1+2 — `vst1_u8` writes 8 bytes into `tmp` (`[u8; 8]`),
  // in-bounds; `neon` confirmed by the dispatch gate. Only the low 4
  // (the pixel) are copied out.
  unsafe { vst1_u8(tmp.as_mut_ptr(), v) };
  out.copy_from_slice(&tmp[..CHANNELS]);
}

/// NEON vertical convolution. Same per-channel vectorization as
/// [`convolve_axis_neon`] but taps stride by a full intermediate row.
/// Bit-identical to [`convolve_vertical_scalar`].
///
/// # Safety
/// 1. `neon` available at runtime — gated by the caller
///    ([`convolve_vertical`]) on `is_aarch64_feature_detected!("neon")`.
/// 2. `inter.len() == out_w * src_h * 4`, `out.len() == out_w * out_h *
///    4`, `coeffs.bounds.len() == out_h` — asserted by the caller.
///    Combined with `ymin + n <= src_h` from [`precompute_coeffs`], every
///    `inter[base..base+4]` access is in-bounds.
/// 3. Same 4-byte load/store contract as [`convolve_axis_neon`].
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn convolve_vertical_neon(
  inter: &[u8],
  out_w: usize,
  _src_h: usize,
  out: &mut [u8],
  out_h: usize,
  coeffs: &Coeffs,
) {
  use std::arch::aarch64::*;
  let ksize = coeffs.ksize;
  let row_stride = out_w * CHANNELS;
  for oy in 0..out_h {
    let (ymin, n) = coeffs.bounds[oy];
    let taps = &coeffs.weights[oy * ksize..oy * ksize + n];
    let out_row = &mut out[oy * row_stride..(oy + 1) * row_stride];
    for ox in 0..out_w {
      let mut acc = vdupq_n_s32(ROUND_BIAS);
      for (i, &w) in taps.iter().enumerate() {
        let base = (ymin + i) * row_stride + ox * CHANNELS;
        // `base + 4 <= inter.len()` by the window invariant
        // (`ymin + n <= src_h`, Safety clause 2).
        let px4 = [
          inter[base],
          inter[base + 1],
          inter[base + 2],
          inter[base + 3],
        ];
        // SAFETY: clauses 1+3 — see `neon_load_rgba`'s contract.
        let v8 = unsafe { neon_load_rgba(px4) };
        let v16 = vmovl_u8(v8);
        let v16lo = vget_low_u16(v16);
        let v32 = vreinterpretq_s32_u32(vmovl_u16(v16lo));
        let wv = vdupq_n_s32(w);
        acc = vmlaq_s32(acc, v32, wv);
      }
      let shifted = vshrq_n_s32::<{ PRECISION_BITS as i32 }>(acc);
      let u16x4 = vqmovun_s32(shifted);
      let u16x8 = vcombine_u16(u16x4, vdup_n_u16(0));
      let u8x8 = vqmovn_u16(u16x8);
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      // SAFETY: clauses 1+3 — see `neon_store_rgba`'s contract.
      unsafe { neon_store_rgba(u8x8, o) };
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Force-scalar variant of [`resize_rgba8`] (calls the `*_scalar`
  /// kernels directly, bypassing the NEON dispatch). Used only by the
  /// differential test to compare against the dispatched path.
  fn resize_rgba8_scalar(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    filter: Filter,
  ) -> Vec<u8> {
    if filter == Filter::Nearest {
      let dst_len = dst_w * dst_h * CHANNELS;
      return resize_nearest(src, src_w, src_h, dst_w, dst_h, dst_len).unwrap();
    }
    let hc = precompute_coeffs(src_w, dst_w, filter).unwrap();
    let vc = precompute_coeffs(src_h, dst_h, filter).unwrap();
    let mut inter = vec![0u8; src_h * dst_w * CHANNELS];
    let mut dst = vec![0u8; dst_w * dst_h * CHANNELS];
    convolve_axis_scalar(src, src_w, src_h, &mut inter, dst_w, &hc);
    convolve_vertical_scalar(&inter, dst_w, src_h, &mut dst, dst_h, &vc);
    dst
  }

  /// Deterministic pseudo-random RGBA8 source (LCG — no rand dependency).
  fn make_src(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_add(1);
    let mut v = Vec::with_capacity(w * h * CHANNELS);
    for _ in 0..w * h * CHANNELS {
      s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      v.push((s >> 24) as u8);
    }
    v
  }

  #[test]
  fn neon_matches_scalar_across_sizes_and_filters() {
    // Differential: the dispatched path (NEON on aarch64, scalar
    // elsewhere) must produce output BIT-IDENTICAL to the force-scalar
    // path, across sizes straddling the 4-channel vector boundary and
    // both up/down scaling. On a non-aarch64 host this is a scalar-vs-
    // scalar identity (still a useful determinism check); on aarch64 it
    // is the real NEON-vs-scalar guarantee.
    let filters = [
      Filter::Bilinear,
      Filter::Bicubic,
      Filter::Lanczos3,
      Filter::Nearest,
    ];
    // Sizes chosen to straddle odd/even widths + up/down + 1-px axes.
    let cases = [
      (4usize, 4usize, 2usize, 2usize),
      (3, 5, 7, 2),
      (5, 3, 2, 8),
      (8, 6, 4, 3),
      (2, 2, 9, 9),
      (5, 1, 2, 1),
      (1, 5, 1, 2),
      (7, 7, 7, 7),
      (16, 9, 5, 11),
    ];
    for (i, &(sw, sh, dw, dh)) in cases.iter().enumerate() {
      let src = make_src(sw, sh, i as u32 * 7 + 1);
      for &f in &filters {
        let dispatched = resize_rgba8(&src, sw, sh, dw, dh, f).unwrap();
        let scalar = resize_rgba8_scalar(&src, sw, sh, dw, dh, f);
        assert_eq!(
          dispatched, scalar,
          "NEON-vs-scalar differential mismatch for {f:?} {sw}x{sh}->{dw}x{dh}"
        );
      }
    }
  }

  #[test]
  fn rejects_zero_dimensions() {
    let src = [0u8; 4]; // 1x1 RGBA
    for (sw, sh, dw, dh) in [(0, 1, 2, 2), (1, 0, 2, 2), (1, 1, 0, 2), (1, 1, 2, 0)] {
      let r = resize_rgba8(
        &src[..sw.max(1) * sh.max(1) * CHANNELS],
        sw,
        sh,
        dw,
        dh,
        Filter::Bilinear,
      );
      assert!(
        matches!(r, Err(Error::ShapeMismatch { .. })),
        "zero dim {sw}x{sh}->{dw}x{dh} must be ShapeMismatch, got {r:?}"
      );
    }
  }

  #[test]
  fn rejects_src_buffer_length_mismatch() {
    // src buffer too short for the claimed dims -> ShapeMismatch (not a
    // panic / OOB read).
    let src = [0u8; 4]; // claims 4 bytes but we say 2x2 (needs 16)
    let r = resize_rgba8(&src, 2, 2, 1, 1, Filter::Bilinear);
    assert!(matches!(r, Err(Error::ShapeMismatch { .. })), "got {r:?}");
  }

  #[test]
  fn rejects_overflowing_dst_product() {
    // dst_w * dst_h * 4 overflows usize -> ShapeMismatch (the structural
    // try_reserve guard's overflow branch). Use usize::MAX-ish dims.
    let src = [0u8; 4];
    let big = usize::MAX / 2 + 1;
    let r = resize_rgba8(&src, 1, 1, big, big, Filter::Bilinear);
    assert!(matches!(r, Err(Error::ShapeMismatch { .. })), "got {r:?}");
  }

  #[test]
  fn output_length_is_exact() {
    // Every accepted resize returns exactly dst_w*dst_h*4 bytes — the
    // invariant `vlm::image::resize` relies on for `ImageBuffer::from_raw`.
    let src = make_src(8, 6, 3);
    for f in [
      Filter::Nearest,
      Filter::Bilinear,
      Filter::Bicubic,
      Filter::Lanczos3,
    ] {
      let out = resize_rgba8(&src, 8, 6, 5, 4, f).unwrap();
      assert_eq!(out.len(), 5 * 4 * CHANNELS, "filter {f:?} output length");
    }
  }

  #[test]
  fn constant_image_is_preserved() {
    // A constant-color image must reproduce the constant at every output
    // pixel for every convolution filter (kernel sums to 1.0). Exact for
    // the integer path (no rounding drift on a flat field).
    let mut src = Vec::with_capacity(6 * 6 * CHANNELS);
    for _ in 0..6 * 6 {
      src.extend_from_slice(&[123, 45, 200, 255]);
    }
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      for &(dw, dh) in &[(3usize, 3usize), (9, 9), (4, 7)] {
        let out = resize_rgba8(&src, 6, 6, dw, dh, f).unwrap();
        for px in out.chunks_exact(CHANNELS) {
          assert_eq!(
            px,
            &[123, 45, 200, 255],
            "constant must survive {f:?} -> {dw}x{dh}"
          );
        }
      }
    }
  }

  #[test]
  fn precompute_coeffs_weights_sum_to_unity_fixedpoint() {
    // Each output index's normalized fixed-point taps should sum to
    // approximately 1<<PRECISION_BITS (the rounding may shift the sum by
    // at most `n` LSB across `n` taps). This guards the normalization.
    let one = 1i64 << PRECISION_BITS;
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      for &(insz, outsz) in &[(8usize, 3usize), (3, 8), (5, 5), (16, 4)] {
        let c = precompute_coeffs(insz, outsz, f).unwrap();
        for o in 0..outsz {
          let (_, n) = c.bounds[o];
          let s: i64 = c.weights[o * c.ksize..o * c.ksize + n]
            .iter()
            .map(|&w| i64::from(w))
            .sum();
          let tol = n as i64 + 1;
          assert!(
            (s - one).abs() <= tol,
            "{f:?} {insz}->{outsz} out {o}: tap sum {s} not within {tol} of {one}"
          );
        }
      }
    }
  }
}
