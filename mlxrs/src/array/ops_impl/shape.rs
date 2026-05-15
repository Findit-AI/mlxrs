//! Method-form shape bridges.

use crate::{array::Array, error::Result, shape::IntoShape};

impl Array {
  /// Reshape this array to the new `shape`. See [`crate::ops::shape::reshape`].
  pub fn reshape(&self, shape: &impl IntoShape) -> Result<Array> {
    crate::ops::shape::reshape(self, shape)
  }
}
