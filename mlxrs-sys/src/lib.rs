//! Raw FFI bindings for mlx-c. Pre-committed bindgen output.
//!
//! Regenerate with `cargo run -p xtask -- regen-bindings` (requires
//! `LIBCLANG_PATH=$(xcode-select -p)/usr/lib`).
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![allow(clippy::missing_safety_doc, clippy::all)]

include!("generated/bindings.rs");

#[cfg(test)]
mod smoke {
  use super::*;
  use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
  };

  // mlx-c's default handler is `printf("MLX error: %s\n", msg); exit(-1);`
  // which would terminate the test process. Install a no-op handler first.
  extern "C" fn noop_handler(_msg: *const c_char, _data: *mut c_void) {}

  #[test]
  fn version_round_trip() {
    unsafe {
      mlx_set_error_handler(Some(noop_handler), ptr::null_mut(), None);
      let mut s = mlx_string_new();
      assert_eq!(mlx_version(&mut s), 0, "mlx_version returned non-zero");
      let data = mlx_string_data(s);
      assert!(!data.is_null());
      let ver = CStr::from_ptr(data).to_string_lossy();
      assert!(!ver.is_empty());
      assert_eq!(mlx_string_free(s), 0);
    }
  }

  #[test]
  fn array_new_free_round_trip() {
    unsafe {
      mlx_set_error_handler(Some(noop_handler), ptr::null_mut(), None);
      let arr = mlx_array_new();
      assert_eq!(mlx_array_free(arr), 0);
    }
  }
}
