//! Base Rotary Position Embedding (RoPE).
//!
//! A 1:1 port of mlx's base RoPE: the python `mlx.nn.RoPE` Module
//! (`python/mlx/nn/layers/positional_encoding.py`) and the swift `RoPE` /
//! `MLXFast.RoPE` pair. Both are thin wrappers over the fused
//! `mlx.fast.rope` primitive (`mlx_fast_rope`), which rotates the first
//! `dims` features of the last axis by an angle proportional to each
//! token's position.
//!
//! For the math see [RoFormer: Enhanced Transformer with Rotary Position
//! Embedding](https://arxiv.org/abs/2104.09864). Two rotation layouts:
//!
//! - **non-traditional** (default, the efficient layout): pairs feature `k`
//!   with feature `k + dims/2`.
//! - **traditional**: rotates consecutive pairs `(2k, 2k+1)`.
//!
//! `offset` shifts every position by a constant — this is the
//! incremental-decoding hook (swift's `applyRotaryPosition(_:to:offset:)`,
//! mlx-lm's `rope(x, offset=cache.offset)`): during single-token decode the
//! query/key for absolute position `p` is fed as a length-1 sequence with
//! `offset = p`, so it is rotated by `p` rather than `0`.
//!
//! `base` and `scale` mirror the reference: `base` is the angular-frequency
//! base (default `10000`), `scale` multiplies the position (a `< 1` scale is
//! linear position-interpolation context extension).
//!
//! # Scope
//!
//! This is the **base** RoPE only. The scaled variants (Llama3 / Su-scaled /
//! YaRN) precompute a per-dimension `freqs` array and pass it to the same
//! primitive with `base = None` — per mlx's contract *exactly one of `base`
//! and `freqs` is `None`* (`python/src/fast.cpp`). They are separate
//! follow-ups; this module always takes the `base` path (`freqs = None`).

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// mlx's default angular-frequency base (`mlx.nn.RoPE`'s `base=10000`).
pub const DEFAULT_BASE: f32 = 10000.0;

/// Apply base rotary position embedding to `x`, rotating its first `dims`
/// features. Free-fn mirror of python `mx.fast.rope(x, dims, traditional=,
/// base=, scale=, offset=)` (the body of `mlx.nn.RoPE.__call__`) and swift
/// `MLXFast.RoPE`.
///
/// - `x`: input array; RoPE rotates over the last axis. Any leading batch /
///   head / sequence dims are preserved (mlx-lm feeds `[B, n_heads, S,
///   head_dim]`). If `head_dim > dims` the trailing `head_dim - dims`
///   features pass through unchanged.
/// - `dims`: number of leading features of the last axis to rotate (must be
///   even and `<= head_dim`; mlx validates and surfaces a recoverable error
///   otherwise).
/// - `traditional`: `true` rotates consecutive pairs `(2k, 2k+1)`; `false`
///   (the default) pairs `k` with `k + dims/2` (more efficient).
/// - `base`: angular-frequency base (mlx default [`DEFAULT_BASE`]).
/// - `scale`: position scale (`1.0` = identity; `< 1.0` = linear
///   position-interpolation).
/// - `offset`: constant added to every position — the KV-cache decode hook
///   (rotate a length-1 step as if at absolute position `offset`).
///
/// Returns a new array the same shape/dtype as `x`. Does **not** evaluate;
/// like every `mlxrs` op it appends to the lazy graph (eval is an explicit
/// `&mut` step on the result).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html).
pub fn rope(
  x: &Array,
  dims: i32,
  traditional: bool,
  base: f32,
  scale: f32,
  offset: i32,
) -> Result<Array> {
  // Base path: `base` is present and `freqs` is the null handle. mlx's
  // contract is "exactly one of `base` and `freqs` must be None"
  // (python/src/fast.cpp); the base RoPE always supplies `base`.
  let base_opt = mlxrs_sys::mlx_optional_float {
    value: base,
    has_value: true,
  };
  // SAFETY: `mlx_array_new()` returns a fresh empty handle (NULL ctx) per the
  // mlx-c convention. It is wrapped in the RAII newtype so it is freed on
  // drop; a NULL-ctx `mlx_array` *is* the absent-optional `freqs` value mlx-c
  // accepts, and the guard keeps it alive across the FFI call below.
  let null_freqs = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `mlx_array_new()` yields a fresh empty out-param handle (NULL ctx);
  // it is wrapped in the RAII newtype FIRST so an early return / panic frees
  // it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles, live for the
  // call and not retained by mlx past it — `x.0` is the input and `null_freqs.0`
  // is the NULL-ctx placeholder mlx-c accepts for the absent `freqs` (kept
  // alive by `null_freqs`); the out-param was freshly allocated above and is
  // written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fast_rope(
      &mut out.0,
      x.0,
      dims,
      traditional,
      base_opt,
      scale,
      offset,
      null_freqs.0,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Base rotary position embedding as a reusable config, mirroring the python
/// `mlx.nn.RoPE` Module and swift `RoPE` layer: it holds the fixed
/// `dims` / `traditional` / `base` / `scale` and is applied per-step with an
/// [`apply`](Rope::apply) call that takes only the position `offset`.
///
/// This is the shape attention layers store once (`self.rope = Rope::new(..)`)
/// and call as `self.rope.apply(&queries, cache_offset)?` each forward pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rope {
  /// Number of leading features of the last axis to rotate.
  pub dims: i32,
  /// Rotation layout: `true` = consecutive pairs `(2k, 2k+1)`; `false`
  /// (default) = `k` paired with `k + dims/2`.
  pub traditional: bool,
  /// Angular-frequency base (mlx default [`DEFAULT_BASE`]).
  pub base: f32,
  /// Position scale (`1.0` = identity).
  pub scale: f32,
}

impl Rope {
  /// Construct a RoPE config. Mirrors `mlx.nn.RoPE(dims, traditional, base,
  /// scale)` with mlx's defaults (`traditional=False`, `base=10000`,
  /// `scale=1.0`) — see [`Rope::standard`] for the all-defaults shorthand.
  pub fn new(dims: i32, traditional: bool, base: f32, scale: f32) -> Self {
    Self {
      dims,
      traditional,
      base,
      scale,
    }
  }

  /// RoPE with mlx's defaults (`traditional=false`, `base=10000`,
  /// `scale=1.0`) — the common case, matching `mlx.nn.RoPE(dims)`.
  pub fn standard(dims: i32) -> Self {
    Self::new(dims, false, DEFAULT_BASE, 1.0)
  }

  /// Apply this RoPE to `x` at position `offset`. Mirrors
  /// `mlx.nn.RoPE.__call__(x, offset=offset)` / swift `RoPE`'s
  /// `callAsFunction(_:offset:)`: `offset = 0` for a full prompt, the KV-cache
  /// offset for incremental decode. Returns a new lazy array (no eval).
  pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
    rope(
      x,
      self.dims,
      self.traditional,
      self.base,
      self.scale,
      offset,
    )
  }
}

#[cfg(test)]
// Golden RoPE outputs are written at 7 significant digits for readability and
// compared with a `1e-5` tolerance (see `TOL`); a few land a digit past f32's
// resolution. The extra digit is intentional documentation of the reference
// value, not a real-precision claim, so the lint is silenced module-wide.
#[allow(clippy::excessive_precision)]
mod tests {
  use super::*;

  /// Golden values were derived directly from the canonical RoPE formula
  /// (`out[a] = x[a]*cos(θ) - x[b]*sin(θ)`,
  /// `out[b] = x[b]*cos(θ) + x[a]*sin(θ)` with
  /// `θ = (offset + n) * scale * base^(-2i/dims)`), evaluated in f64 and
  /// rounded to 7 digits. Pairs are `(k, k+dims/2)` (non-traditional) or
  /// `(2k, 2k+1)` (traditional). A wider-than-needed `1e-5` abs tolerance
  /// absorbs the f32-vs-f64 / fused-kernel rounding gap.
  const TOL: f32 = 1e-5;

  /// `[1, 1, 2, 4]` input `[[0,1,2,3],[4,5,6,7]]` — two tokens, head_dim 4.
  fn input() -> Array {
    Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &(1, 1, 2, 4)).unwrap()
  }

  fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
      assert!(
        (g - w).abs() <= TOL,
        "index {i}: got {g}, want {w} (|Δ|={})",
        (g - w).abs()
      );
    }
  }

  #[test]
  fn non_traditional_offset0() {
    let x = input();
    let mut y = rope(&x, 4, false, DEFAULT_BASE, 1.0, 0).unwrap();
    // Token 0 (position 0): θ=0 ⇒ cos=1, sin=0 ⇒ identity.
    assert_close(
      &y.to_vec::<f32>().unwrap(),
      &[
        0.0, 1.0, 2.0, 3.0, // token 0 unchanged
        -2.8876167, 4.9297512, 6.6076978, 7.0496492, // token 1
      ],
    );
  }

  #[test]
  fn non_traditional_offset2() {
    let x = input();
    // offset=2 ⇒ token 0 is at position 2, token 1 at position 3.
    let mut y = rope(&x, 4, false, DEFAULT_BASE, 1.0, 2).unwrap();
    assert_close(
      &y.to_vec::<f32>().unwrap(),
      &[
        -1.8185949, 0.9398040, -0.8322937, 3.0193987, // token 0 @ pos 2
        -4.8066900, 4.7877817, -5.3754749, 7.1468277, // token 1 @ pos 3
      ],
    );
  }

  #[test]
  fn traditional_offset0() {
    let x = input();
    let mut y = rope(&x, 4, true, DEFAULT_BASE, 1.0, 0).unwrap();
    assert_close(
      &y.to_vec::<f32>().unwrap(),
      &[
        0.0, 1.0, 2.0, 3.0, // token 0 unchanged
        -2.0461457, 6.0673955, 5.9297012, 7.0596490, // token 1
      ],
    );
  }

  #[test]
  fn traditional_offset2() {
    let x = input();
    let mut y = rope(&x, 4, true, DEFAULT_BASE, 1.0, 2).unwrap();
    assert_close(
      &y.to_vec::<f32>().unwrap(),
      &[
        -0.9092974, -0.4161468, 1.9396040, 3.0393974, // token 0 @ pos 2
        -4.6655700, -4.3854825, 5.7873317, 7.1768232, // token 1 @ pos 3
      ],
    );
  }

  #[test]
  fn scale_half_is_position_interpolation() {
    let x = input();
    // scale=0.5 ⇒ token 1 is rotated as if at position 0.5.
    let mut y = rope(&x, 4, false, DEFAULT_BASE, 0.5, 0).unwrap();
    assert_close(
      &y.to_vec::<f32>().unwrap(),
      &[
        0.0, 1.0, 2.0, 3.0, // token 0 (pos 0) unchanged
        0.6337770, 4.9649376, 7.1831975, 7.0249124, // token 1 (pos 0.5)
      ],
    );
  }

  #[test]
  fn partial_dims_pass_through_tail() {
    let x = input();
    // dims=2 (< head_dim 4): rotate only the first 2 features; the trailing
    // two pass through unchanged.
    let mut y = rope(&x, 2, false, DEFAULT_BASE, 1.0, 0).unwrap();
    assert_close(
      &y.to_vec::<f32>().unwrap(),
      &[
        0.0, 1.0, 2.0, 3.0, // token 0 unchanged
        -2.0461457, 6.0673955, 6.0, 7.0, // token 1: [2,3] rotated, [6,7] kept
      ],
    );
  }

  #[test]
  fn config_apply_matches_free_fn() {
    let x = input();
    let r = Rope::new(4, false, DEFAULT_BASE, 1.0);
    let mut via_config = r.apply(&x, 2).unwrap();
    let mut via_fn = rope(&x, 4, false, DEFAULT_BASE, 1.0, 2).unwrap();
    assert_close(
      &via_config.to_vec::<f32>().unwrap(),
      &via_fn.to_vec::<f32>().unwrap(),
    );
  }

  #[test]
  fn standard_uses_mlx_defaults() {
    let r = Rope::standard(8);
    assert_eq!(r.dims, 8);
    assert!(!r.traditional);
    assert_eq!(r.base, DEFAULT_BASE);
    assert_eq!(r.scale, 1.0);
  }
}
