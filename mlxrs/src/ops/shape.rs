//! Shape ops: reshape (Phase 3.5 archetype #3 — IntoShape pattern). Phase 4
//! fills in transpose/expand_dims/squeeze/etc.

use crate::{
  array::Array,
  error::{Result, check},
  shape::IntoShape,
  stream::default_stream,
};

/// Reshape `a` to a new shape. Errors on incompatible total element count
/// (the C++ side validates).
///
/// CANONICAL SHAPE ARCHETYPE — the `IntoShape::with_shape` callback pattern
/// used by every shape-taking op. Every reshape/expand_dims/squeeze/etc.
/// follows this exact shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html).
pub fn reshape(a: &Array, shape: &impl IntoShape) -> Result<Array> {
  shape.with_shape(|s| {
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    check(unsafe {
      mlxrs_sys::mlx_reshape(&mut out.0, a.0, s.as_ptr(), s.len(), default_stream())
    })?;
    Ok(out)
  })
}
