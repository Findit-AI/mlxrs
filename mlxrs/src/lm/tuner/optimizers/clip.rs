//! [`clip_grad_norm`] — global-norm gradient clipping. Placeholder.

#![allow(dead_code)]

use crate::{Array, Result, lm::load::Weights};

/// Stub.
pub fn clip_grad_norm(_grads: &mut Weights, _max_norm: f32) -> Result<Array> {
  Err(crate::error::Error::Backend {
    message: "clip_grad_norm: not yet implemented".into(),
  })
}
