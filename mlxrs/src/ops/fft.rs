//! FFT ops: forward/inverse 1-D, 2-D, N-D, real-input variants, and shifts.
//!
//! All FFT ops accept an [`FftNorm`] strategy. The default in mlx-python is
//! `FftNorm::Backward` (no scaling on the forward, `1/N` on the inverse). The
//! one-axis ops also accept an `n` length (for zero-pad/truncate to a target
//! transform length) and an `axis` index.
//!
//! Multi-axis ops (`fft2`, `fftn`, etc.) take parallel `n` and `axes` slices.
//! Empty `axes` is treated as "all axes" by mlx-python's `fftn`/`ifftn`/etc.;
//! we route empty slices through `dim_ptr`'s static sentinel rather than the
//! Rust dangling pointer for empty `&[i32]`.
//!
//! See [mlx FFT docs](https://ml-explore.github.io/mlx/build/html/python/fft.html).

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Result, check},
  shape::dim_ptr,
  stream::default_stream,
};

/// Normalization mode for FFT ops. Mirrors `mlx.core.fft`'s `norm=` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FftNorm {
  /// No scaling on forward, `1/N` on inverse. Matches numpy/mlx-python default.
  #[default]
  Backward,
  /// `1/sqrt(N)` on both forward and inverse (unitary FFT).
  Ortho,
  /// `1/N` on forward, no scaling on inverse.
  Forward,
}

impl From<FftNorm> for mlxrs_sys::mlx_fft_norm {
  fn from(n: FftNorm) -> Self {
    match n {
      FftNorm::Backward => mlxrs_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
      FftNorm::Ortho => mlxrs_sys::mlx_fft_norm__MLX_FFT_NORM_ORTHO,
      FftNorm::Forward => mlxrs_sys::mlx_fft_norm__MLX_FFT_NORM_FORWARD,
    }
  }
}

/// 1-D discrete Fourier transform along `axis`. `n` is the transform length
/// (zero-pad or truncate `a` along `axis` to this length before transforming).
/// Output dtype is Complex64.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft.html).
pub fn fft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_fft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D inverse discrete Fourier transform along `axis`. See [`fft`] for the
/// semantics of `n` and `norm`. Output dtype is Complex64.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifft.html).
pub fn ifft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_ifft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D real-input FFT along `axis`. Input is real-valued; output is complex
/// with the redundant negative-frequency half dropped (length `n/2 + 1`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft.html).
pub fn rfft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_rfft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D inverse of [`rfft`]: complex-valued one-sided spectrum -> real signal.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfft.html).
pub fn irfft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_irfft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D FFT over the listed `axes` with per-axis transform lengths `n`. Empty
/// `axes` is routed through `dim_ptr`'s static sentinel so the FFI never sees
/// a Rust dangling pointer.
///
/// `n` and `axes` must have the same length when `axes` is non-empty (mlx-c
/// validates and surfaces the error).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftn.html).
pub fn fftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_fftn(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D inverse of [`fftn`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifftn.html).
pub fn ifftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_ifftn(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D FFT over the last two axes by default; pass `axes` to override.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft2.html).
pub fn fft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_fft2(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D inverse of [`fft2`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifft2.html).
pub fn ifft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_ifft2(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D real-input FFT.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfftn.html).
pub fn rfftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_rfftn(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D inverse of [`rfftn`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfftn.html).
pub fn irfftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_irfftn(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D real-input FFT (last two axes by default).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft2.html).
pub fn rfft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_rfft2(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D inverse of [`rfft2`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfft2.html).
pub fn irfft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_irfft2(
      &mut out.0,
      a.0,
      dim_ptr(n),
      n.len(),
      dim_ptr(axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Sample frequencies for [`fft`] of length `n` and sample spacing `d`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftfreq.html).
pub fn fftfreq(n: i32, d: f64) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_fft_fftfreq(&mut out.0, n as c_int, d, default_stream()) })?;
  Ok(out)
}

/// Sample frequencies for [`rfft`] of length `n` and sample spacing `d`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfftfreq.html).
pub fn rfftfreq(n: i32, d: f64) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_fft_rfftfreq(&mut out.0, n as c_int, d, default_stream()) })?;
  Ok(out)
}

/// Shift the zero-frequency component to the center along the listed `axes`.
/// Empty `axes` shifts all axes (mlx-python default), routed through the
/// `dim_ptr` sentinel for FFI safety.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftshift.html).
pub fn fftshift(a: &Array, axes: &[i32]) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_fftshift(&mut out.0, a.0, dim_ptr(axes), axes.len(), default_stream())
  })?;
  Ok(out)
}

/// Inverse of [`fftshift`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifftshift.html).
pub fn ifftshift(a: &Array, axes: &[i32]) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_fft_ifftshift(&mut out.0, a.0, dim_ptr(axes), axes.len(), default_stream())
  })?;
  Ok(out)
}
