//! Safe `mlx_closure` wrapper.
//!
//! `mlx_closure` is mlx-c's callable handle: a function-pointer + opaque
//! `void* payload` pair that the autograd / custom-VJP / checkpoint / compile
//! transforms accept as their user-supplied function argument. The trampoline
//! pattern here mirrors `mlx-swift`'s
//! [`new_mlx_closure`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Cmlx%2BUtil.swift)
//! (`Cmlx+Util.swift`) and the equivalent `pybind` shim on the Python side.
//!
//! ## Lifetime
//!
//! The Rust callable is boxed (`Box<Inner<F>>`) and `Box::into_raw`'d into a
//! stable `*mut c_void` payload pointer. A C destructor (`destroy_payload`)
//! reclaims the box via `Box::from_raw`. `mlx_closure_free` invokes the
//! destructor exactly once (mlx-c's `mlx_closure` is a shared_ptr-backed
//! handle, so dtor runs when the last reference drops, not necessarily at the
//! `mlx_closure_free` call). The [`Closure`] wrapper owns *one* reference to
//! the handle and frees it in [`Drop`]; the payload box is *not* owned by
//! [`Closure`] directly — it is owned by the C++ shared destructor.
//!
//! ## Re-entrancy and panics
//!
//! The trampoline catches Rust panics via [`std::panic::catch_unwind`] and
//! converts them to a non-zero rc — unwinding across the `extern "C"` boundary
//! is undefined behavior. The user function is required `Fn + 'static` (not
//! `FnMut`); aliasing the captured state across re-entrant mlx-c calls is
//! safe because `Fn` mandates `&self` access.

use std::{
  ffi::c_void,
  os::raw::c_int,
  panic::{AssertUnwindSafe, catch_unwind},
  ptr,
};

use crate::{
  Array,
  error::{Error, Result, ensure_handler_installed},
};

/// Boxed type-erased Rust callable invoked by the mlx-c trampoline.
///
/// `Box<dyn Fn(&[Array]) -> Result<Vec<Array>>>` is itself a fat pointer
/// (vtable + data), so we wrap it in an outer `Box` to land on a stable
/// thin `*mut c_void` (the inner `Box<dyn Fn>` already heap-allocates the
/// closure; the outer `Box` is the indirection layer mlx-c hands back).
pub(crate) type BoxedFn = Box<dyn Fn(&[Array]) -> Result<Vec<Array>> + 'static>;

/// Safe RAII wrapper around an `mlx_closure` that keeps the captured Rust
/// callable alive for the entire lifetime of the C handle.
///
/// Construct via [`Closure::new`]; the returned value owns one reference to
/// the underlying `mlx_closure` and frees it on [`Drop`]. To pass the handle
/// into mlx-c transforms (`mlx_value_and_grad`, `mlx_vjp`, …) use
/// [`Closure::as_raw`], which borrows the handle without transferring
/// ownership. The Rust callable is held alive by the closure's mlx-c
/// destructor, *not* by this struct.
///
/// `Closure` is intentionally `!Send` + `!Sync`: the captured `F` may
/// reference [`crate::Array`] handles (themselves `!Send`), and the mlx-c
/// closure's payload destructor must run on the thread that built it.
pub struct Closure {
  inner: mlxrs_sys::mlx_closure,
}

impl Closure {
  /// Construct a closure from a Rust callable. Returns `Err` if the underlying
  /// `mlx_closure_new_func_payload` allocation fails.
  ///
  /// `F` is required `Fn + 'static` so the mlx-c side can invoke it across
  /// arbitrary later re-entries (including from within `mlx_eval`).
  pub fn new<F>(f: F) -> Result<Self>
  where
    F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
  {
    ensure_handler_installed();
    // Box the user closure on the heap, then re-box the resulting fat trait-
    // object pointer so the payload we hand to C is a thin `*mut c_void`.
    // SAFETY of pointer round-trip: we recover the same `Box<BoxedFn>` via
    // `Box::from_raw` exactly once, in `destroy_payload`. mlx-c invokes the
    // destructor exactly once when the underlying `shared_ptr` reaches
    // refcount 0.
    let boxed: Box<BoxedFn> = Box::new(Box::new(f));
    let payload_ptr: *mut c_void = Box::into_raw(boxed).cast();

    // SAFETY: `trampoline::<F>` and `destroy_payload` are both `extern "C"`
    // with the exact signatures mlx-c expects. `payload_ptr` is a freshly
    // boxed `Box<BoxedFn>` whose lifetime is transferred to mlx-c by this
    // call: mlx-c IMMEDIATELY wraps it in `std::shared_ptr<void>(payload,
    // dtor)` as the very first statement of its `try` block (see vendored
    // `mlx-c/mlx/c/closure.cpp::mlx_closure_new_func_payload`, line 70).
    // From that point on, the shared_ptr OWNS the payload — even if any
    // later allocation inside the same `try` throws (e.g. the lambda
    // capture or `mlx_closure_new_(cpp_closure)`), the shared_ptr's
    // destructor runs `destroy_payload(payload_ptr)` as part of stack
    // unwinding before the `catch` clause returns a NULL closure to us.
    // Therefore the NULL-ctx return path below MUST NOT reclaim the box
    // ourselves — that would double-free / UAF.
    let inner = unsafe {
      mlxrs_sys::mlx_closure_new_func_payload(Some(trampoline), payload_ptr, Some(destroy_payload))
    };
    if inner.ctx.is_null() {
      // mlx-c already owns `payload_ptr` via the
      // `std::shared_ptr<void>(payload, dtor)` it constructed at the top
      // of its `try` block. If the C++ ctor threw post-shared_ptr-
      // construction, the shared_ptr destructor has ALREADY released the
      // payload via `destroy_payload`. Reclaiming with `Box::from_raw`
      // here would be a double-free / UAF.
      //
      // We accept the (tiny) leak on the alternate path where mlx-c
      // returns NULL without ever constructing the shared_ptr (i.e. the
      // `mlx_closure_new_()` infallible sentinel constructor on the
      // catch arm somehow surfaced NULL — not currently observed in any
      // mlx-c codepath but a defensive consideration). Leak is strictly
      // preferable to UAF.
      return Err(crate::error::take_last().unwrap_or(Error::Backend {
        message: "mlx_closure_new_func_payload returned NULL ctx".into(),
      }));
    }
    Ok(Self { inner })
  }

  /// Borrow the raw `mlx_closure` handle for a transient FFI call.
  ///
  /// The returned handle MUST NOT be retained past this `&self` borrow —
  /// `Drop` will free the underlying handle. mlx-c transforms that consume a
  /// closure by *value* internally take a shared_ptr copy, so passing
  /// `closure.as_raw()` into e.g. `mlx_value_and_grad` is sound.
  #[inline]
  pub fn as_raw(&self) -> mlxrs_sys::mlx_closure {
    self.inner
  }
}

impl Drop for Closure {
  fn drop(&mut self) {
    // SAFETY: frees the handle this `Closure` owns exactly once. The closure's
    // C++ shared_ptr refcount drops; when it hits 0 the payload destructor
    // we registered runs and reclaims the `Box<BoxedFn>`. Runs during `Drop`
    // so must not touch TLS / panic / unwind across `extern "C"` — the rc is
    // discarded silently per the crate's `Drop` convention.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_free(self.inner);
    }
  }
}

// ─────────────────────────── trampoline ───────────────────────────

/// `extern "C"` shim invoked by mlx-c whenever the closure is applied.
///
/// `outputs_out` is an out-parameter slot pre-allocated by the caller (NULL
/// `ctx`); we populate it via `mlx_vector_array_set_data`. `inputs` is owned
/// by mlx-c (we read it; we do NOT free it). `payload` is the `*mut c_void`
/// we registered.
///
/// Returns `0` on success, non-zero on user error or panic. On user error /
/// panic we leave `outputs_out` populated with an empty `mlx_vector_array`
/// (still a valid handle that mlx-c will free) and post a `Backend` message
/// into the TLS error slot so `crate::error::check(rc)` can drain it.
extern "C" fn trampoline(
  outputs_out: *mut mlxrs_sys::mlx_vector_array,
  inputs: mlxrs_sys::mlx_vector_array,
  payload: *mut c_void,
) -> c_int {
  // Wrap the entire body in `catch_unwind` — any panic across `extern "C"`
  // is UB. We restore the panic as a Backend error in the TLS slot.
  let result = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: `payload` is the `*mut c_void` we stored via `Box::into_raw`
    // (preserved by mlx-c across calls). We cast back to `*const BoxedFn` and
    // borrow — NOT take ownership; the box is reclaimed in `destroy_payload`.
    let f: &BoxedFn = unsafe { &*payload.cast::<BoxedFn>() };

    // Borrow the input handles WITHOUT taking ownership: we build a
    // `Vec<Array>` of fresh handles by copying each element via
    // `mlx_vector_array_get` (refcount bump) — the original `inputs` vector
    // is mlx-c's. We then call the user function with a `&[Array]` borrow.
    let inputs_vec = borrow_inputs(inputs)?;

    // Invoke user function.
    let outputs = f(&inputs_vec)?;

    // Marshal outputs back into the out-param `mlx_vector_array`. We use
    // `mlx_vector_array_set_data` which copies the array handles into the
    // existing vector slot (refcount bump on each).
    write_outputs(outputs_out, &outputs)?;
    Ok::<(), Error>(())
  }));
  match result {
    Ok(Ok(())) => 0,
    Ok(Err(e)) => {
      // Stash the user error in TLS so `check(rc)` drains it.
      crate::error::set_last(e);
      // Populate the out-param with an empty vector so mlx-c's later
      // `mlx_vector_array_free` is a defined no-op.
      // SAFETY: `outputs_out` is the caller-owned pre-allocated handle slot;
      // writing an empty vector handle is the safe way to leave it.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
    Err(panic_payload) => {
      let msg = if let Some(s) = panic_payload.downcast_ref::<&'static str>() {
        (*s).to_string()
      } else if let Some(s) = panic_payload.downcast_ref::<String>() {
        s.clone()
      } else {
        "panic in mlxrs::transforms closure trampoline".to_string()
      };
      crate::error::set_last(Error::Backend {
        message: format!("mlxrs::transforms closure trampoline caught panic: {msg}"),
      });
      // SAFETY: same as above — leave the out-param holding an empty handle.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
  }
}

/// `extern "C"` destructor mlx-c invokes when the closure's last `shared_ptr`
/// copy drops. Reclaims the `Box<BoxedFn>` we leaked at construction.
extern "C" fn destroy_payload(payload: *mut c_void) {
  if payload.is_null() {
    return;
  }
  // SAFETY: `payload` is the `*mut c_void` produced by `Box::into_raw` on a
  // `Box<BoxedFn>` in `Closure::new`. mlx-c calls this destructor exactly
  // once per registration. Box ownership returns here and is dropped.
  // Wrap drop in `catch_unwind` so a panicking user closure-destructor
  // cannot unwind across the C++ boundary.
  let _ = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: see fn doc above — payload is a Box<BoxedFn> we created.
    let _: Box<BoxedFn> = unsafe { Box::from_raw(payload.cast::<BoxedFn>()) };
  }));
}

// ─────────────────────── vector_array marshalling ───────────────────────

/// Build a `Vec<Array>` from an mlx-c `mlx_vector_array` by copying out each
/// handle (refcount bump on each via `mlx_array_set`).
pub(crate) fn drain_vector(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does not
  // mutate or retain it and returns a plain length.
  let n = unsafe { mlxrs_sys::mlx_vector_array_size(vec) };
  let mut parts = Vec::with_capacity(n);
  for i in 0..n {
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
    // ctx); wrapping in `Array` first ensures `Drop` reclaims on early return.
    let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: valid `vec` handle; `part.0` is the freshly-allocated out-param
    // populated by this call. rc surfaced via `check()`.
    crate::error::check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, vec, i) })?;
    parts.push(part);
  }
  Ok(parts)
}

/// Borrow the input handles of a `mlx_vector_array` as a `Vec<Array>` of
/// fresh refcount-shared copies. Same effect as [`drain_vector`] but used
/// inside the trampoline where the source `vec` is owned by mlx-c (we MUST
/// NOT free it).
fn borrow_inputs(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  drain_vector(vec)
}

/// Pack a `&[Array]` into a freshly allocated `mlx_vector_array` and write
/// its handle into `out`. mlx-c copies refcount-shared array handles into
/// the new vector storage. The previous contents of `*out` are leaked — mlx-c
/// gives us a NULL-ctx slot on first entry, so this is a safe overwrite.
fn write_outputs(out: *mut mlxrs_sys::mlx_vector_array, outputs: &[Array]) -> Result<()> {
  // Collect raw handles into a contiguous `Vec<mlx_array>` for FFI.
  let raw: Vec<mlxrs_sys::mlx_array> = outputs.iter().map(|a| a.0).collect();
  let data_ptr = if raw.is_empty() {
    ptr::null()
  } else {
    raw.as_ptr()
  };
  // SAFETY: `out` is the trampoline's caller-owned out-param. Per mlx-c's
  // convention on entry it is a NULL-ctx handle; we replace it with a fresh
  // vector populated from `raw` (mlx-c copies the array handles, refcount-
  // bumping each).
  unsafe {
    *out = mlxrs_sys::mlx_vector_array_new_data(data_ptr, raw.len());
  }
  // SAFETY: post-write null-check — the constructor is fallible.
  if unsafe { (*out).ctx.is_null() } && !outputs.is_empty() {
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_vector_array_new_data returned NULL ctx in closure trampoline".into(),
    }));
  }
  Ok(())
}

// ─────────────────────── caller-side helpers ───────────────────────

/// RAII guard for a temporary `mlx_vector_array`. Constructed *before* the
/// populating call so an early return / panic still frees it.
pub(crate) struct VectorArrayGuard(pub(crate) mlxrs_sys::mlx_vector_array);
impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Same `Drop`
    // discipline as elsewhere — discard rc silently.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}

/// Pack a `&[Array]` (or `&[&Array]` via iterator) into a fresh
/// `mlx_vector_array`. Returns the handle wrapped in a guard for RAII free.
pub(crate) fn vector_array_from_borrow(arrays: &[&Array]) -> Result<VectorArrayGuard> {
  ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
  let data_ptr = if raw.is_empty() {
    ptr::null()
  } else {
    raw.as_ptr()
  };
  // SAFETY: `data_ptr` is either NULL (n==0, mlx-c builds an empty vector) or
  // a valid pointer to `raw.len()` borrowed handles live for this call (mlx-c
  // copies into the new vector, refcount-bumping each).
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(data_ptr, raw.len()) };
  if vec.ctx.is_null() {
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_vector_array_new_data returned NULL ctx".into(),
    }));
  }
  Ok(VectorArrayGuard(vec))
}

/// Same as [`vector_array_from_borrow`] but takes `&[Array]` (most-common
/// caller convenience).
pub(crate) fn vector_array_from_slice(arrays: &[Array]) -> Result<VectorArrayGuard> {
  ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
  let data_ptr = if raw.is_empty() {
    ptr::null()
  } else {
    raw.as_ptr()
  };
  // SAFETY: `data_ptr` is either NULL or a valid pointer to `raw.len()`
  // borrowed handles live for this call; mlx-c copies into the new vector.
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(data_ptr, raw.len()) };
  if vec.ctx.is_null() {
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_vector_array_new_data returned NULL ctx".into(),
    }));
  }
  Ok(VectorArrayGuard(vec))
}

/// RAII guard for a temporary `mlx_closure_value_and_grad`.
pub(crate) struct ClosureValueAndGradGuard(pub(crate) mlxrs_sys::mlx_closure_value_and_grad);
impl Drop for ClosureValueAndGradGuard {
  fn drop(&mut self) {
    // SAFETY: same discipline as `VectorArrayGuard` — single-owner free,
    // rc discarded.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_value_and_grad_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_closure_custom`.
pub(crate) struct ClosureCustomGuard(pub(crate) mlxrs_sys::mlx_closure_custom);
impl Drop for ClosureCustomGuard {
  fn drop(&mut self) {
    // SAFETY: same discipline as `VectorArrayGuard` — single-owner free.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_custom_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_closure` that we own (e.g. the result of
/// `mlx_checkpoint` / `mlx_custom_function`).
pub(crate) struct RawClosureGuard(pub(crate) mlxrs_sys::mlx_closure);
impl Drop for RawClosureGuard {
  fn drop(&mut self) {
    // SAFETY: same discipline as `VectorArrayGuard` — single-owner free.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_free(self.0);
    }
  }
}

/// Build a custom-VJP `mlx_closure_custom` from a Rust 3-input function.
///
/// The contract matches `mlx_custom_vjp`'s `fun_vjp` argument:
/// `(primals, outputs, cotangents) -> grads`.
pub(crate) fn closure_custom_new<F>(f: F) -> Result<ClosureCustomGuard>
where
  F: Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  let boxed: Box<BoxedFn3> = Box::new(Box::new(f));
  let payload_ptr: *mut c_void = Box::into_raw(boxed).cast();
  // SAFETY: trampoline + destructor have correct signatures. `payload_ptr` is
  // a freshly leaked `Box<BoxedFn3>` whose lifetime is transferred to mlx-c
  // by this call: mlx-c IMMEDIATELY wraps it in
  // `std::shared_ptr<void>(payload, dtor)` as the first statement of its
  // `try` block (see vendored
  // `mlx-c/mlx/c/closure.cpp::mlx_closure_custom_new_func_payload`,
  // line 471). From that point on the shared_ptr OWNS the payload — even
  // if any later allocation inside the same `try` throws, the shared_ptr
  // destructor runs `destroy_payload_3(payload_ptr)` during unwinding
  // before the `catch` clause returns NULL. Therefore the NULL-ctx return
  // path below MUST NOT reclaim the box — that would double-free / UAF.
  let inner = unsafe {
    mlxrs_sys::mlx_closure_custom_new_func_payload(
      Some(trampoline_custom),
      payload_ptr,
      Some(destroy_payload_3),
    )
  };
  if inner.ctx.is_null() {
    // mlx-c already owns `payload_ptr` via its `shared_ptr<void>`; the
    // shared_ptr destructor has run (or will run on the natural drop
    // path) and released the payload via `destroy_payload_3`. DO NOT
    // reclaim manually — that would be a double-free / UAF. Same
    // rationale as `Closure::new` above: accept a (tiny) leak on the
    // unobserved-NULL path over a deterministic UAF.
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_closure_custom_new_func_payload returned NULL ctx".into(),
    }));
  }
  Ok(ClosureCustomGuard(inner))
}

pub(crate) type BoxedFn3 =
  Box<dyn Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static>;

extern "C" fn trampoline_custom(
  outputs_out: *mut mlxrs_sys::mlx_vector_array,
  primals: mlxrs_sys::mlx_vector_array,
  outputs: mlxrs_sys::mlx_vector_array,
  cotangents: mlxrs_sys::mlx_vector_array,
  payload: *mut c_void,
) -> c_int {
  let result = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: `payload` was produced by `Box::into_raw(Box<BoxedFn3>)` and
    // is preserved by mlx-c; borrow without taking ownership.
    let f: &BoxedFn3 = unsafe { &*payload.cast::<BoxedFn3>() };
    let p = borrow_inputs(primals)?;
    let o = borrow_inputs(outputs)?;
    let c = borrow_inputs(cotangents)?;
    let grads = f(&p, &o, &c)?;
    write_outputs(outputs_out, &grads)?;
    Ok::<(), Error>(())
  }));
  match result {
    Ok(Ok(())) => 0,
    Ok(Err(e)) => {
      crate::error::set_last(e);
      // SAFETY: leave out-param holding an empty vector handle.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
    Err(panic_payload) => {
      let msg = if let Some(s) = panic_payload.downcast_ref::<&'static str>() {
        (*s).to_string()
      } else if let Some(s) = panic_payload.downcast_ref::<String>() {
        s.clone()
      } else {
        "panic in mlxrs::transforms custom-VJP trampoline".to_string()
      };
      crate::error::set_last(Error::Backend {
        message: format!("mlxrs::transforms custom-VJP trampoline caught panic: {msg}"),
      });
      // SAFETY: leave out-param holding an empty vector handle.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
  }
}

extern "C" fn destroy_payload_3(payload: *mut c_void) {
  if payload.is_null() {
    return;
  }
  let _ = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: payload is a Box<BoxedFn3> we created; reclaim ownership once.
    let _: Box<BoxedFn3> = unsafe { Box::from_raw(payload.cast::<BoxedFn3>()) };
  }));
}
