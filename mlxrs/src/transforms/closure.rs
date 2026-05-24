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
    // In production we call mlx-c directly. In `cfg(test)` builds we route
    // through a swappable function pointer (`test_seam::closure_new_fn`) so
    // unit tests can inject a NULL-returning stub that exercises the
    // `inner.ctx.is_null()` branch where the pre-fix F1 double-free lived —
    // see the `tests::closure_new_returns_err_*` cases in this file. The
    // `#[cfg(test)]` arm defaults to the same FFI symbol; test stubs satisfy
    // the same ABI + ownership contract (see `test_seam` docs), so the
    // unsafe contract is identical between the two arms.
    // SAFETY: `trampoline` and `destroy_payload` have the exact extern "C"
    // signatures mlx-c expects; `payload_ptr` is a freshly leaked
    // `Box<BoxedFn>` whose ownership transfers to mlx-c's shared_ptr per
    // the contract documented above. The `#[cfg(test)]` arm is functionally
    // identical (defaults to the same FFI symbol).
    let inner = unsafe { call_closure_new_ffi(payload_ptr) };
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

// ──────────────────────── FFI call indirection ────────────────────────

/// Invoke `mlx_closure_new_func_payload` (production) or the test-seam stub
/// (`#[cfg(test)]`). Kept in a single helper so the safety annotation lives in
/// exactly one place — see [`Closure::new`] and the `test_seam` docs for the
/// ownership contract on `payload_ptr`.
///
/// # Safety
/// Caller must ensure `payload_ptr` was produced by `Box::into_raw` on a
/// `Box<BoxedFn>` and that ownership is hereby transferred to mlx-c's
/// `shared_ptr<void>(payload, destroy_payload)`. The `#[cfg(test)]` arm
/// routes through a swappable function pointer that defaults to the same
/// FFI symbol; swapped-in stubs must satisfy the identical ABI + ownership
/// contract.
#[inline]
unsafe fn call_closure_new_ffi(payload_ptr: *mut c_void) -> mlxrs_sys::mlx_closure {
  #[cfg(not(test))]
  // SAFETY: forwarded from caller; this is the production direct-FFI arm.
  unsafe {
    mlxrs_sys::mlx_closure_new_func_payload(Some(trampoline), payload_ptr, Some(destroy_payload))
  }
  #[cfg(test)]
  // SAFETY: forwarded from caller; the seam defaults to the same FFI symbol.
  unsafe {
    (test_seam::closure_new_fn())(Some(trampoline), payload_ptr, Some(destroy_payload))
  }
}

/// Invoke `mlx_closure_custom_new_func_payload` (production) or the test-seam
/// stub (`#[cfg(test)]`). Same single-call-site rationale as
/// [`call_closure_new_ffi`].
///
/// # Safety
/// Caller must ensure `payload_ptr` was produced by `Box::into_raw` on a
/// `Box<BoxedFn3>` and that ownership transfers to mlx-c's `shared_ptr`.
#[inline]
unsafe fn call_closure_custom_new_ffi(payload_ptr: *mut c_void) -> mlxrs_sys::mlx_closure_custom {
  #[cfg(not(test))]
  // SAFETY: forwarded from caller; production direct-FFI arm.
  unsafe {
    mlxrs_sys::mlx_closure_custom_new_func_payload(
      Some(trampoline_custom),
      payload_ptr,
      Some(destroy_payload_3),
    )
  }
  #[cfg(test)]
  // SAFETY: forwarded from caller; seam defaults to the same FFI symbol.
  unsafe {
    (test_seam::closure_custom_new_fn())(
      Some(trampoline_custom),
      payload_ptr,
      Some(destroy_payload_3),
    )
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
  // Production: direct FFI; tests: route through the swappable seam so the
  // NULL-ctx branch (which used to double-free in F1) is exercised
  // deterministically by `tests::closure_custom_new_returns_err_*`.
  // SAFETY: `payload_ptr` is a freshly leaked `Box<BoxedFn3>` whose
  // ownership transfers to mlx-c per the contract documented above.
  let inner = unsafe { call_closure_custom_new_ffi(payload_ptr) };
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

// ─────────────────────────── test seam ───────────────────────────

/// Test-only function-pointer indirection over the mlx-c closure constructors.
///
/// Production builds (`#[cfg(not(test))]`) call
/// `mlxrs_sys::mlx_closure_*_new_func_payload` directly: zero indirection,
/// zero overhead. The compiler eliminates this module entirely.
///
/// In `#[cfg(test)]` builds the constructor call in `Closure::new` /
/// `closure_custom_new` routes through a [`std::sync::Mutex`]-protected
/// function pointer here, defaulting to the real mlx-c symbol. The unit
/// tests below swap in a deterministic stub that simulates mlx-c's
/// shared_ptr-then-throw failure mode (invokes the destructor we registered,
/// then returns NULL ctx) to exercise the `inner.ctx.is_null()` branch
/// where the pre-fix F1 double-free lived. Without this seam the NULL-ctx
/// branch is unreachable from Rust (we cannot inject OOM into mlx-c) and
/// CI would be blind to a regression that re-introduced the reclaim.
///
/// Tests acquire the mutex via [`ScopedClosureCtor`] / [`ScopedCustomCtor`]
/// for the duration of a single `Closure::new` / `closure_custom_new`
/// call, restoring the real FFI symbol on drop (panic-safe). The mutex
/// also serializes the seam-test cases — combined with `--test-threads=1`
/// this is belt-and-suspenders against cross-test interference.
#[cfg(test)]
pub(crate) mod test_seam {
  use std::sync::{Mutex, OnceLock};

  use super::*;

  /// Function-pointer type matching `mlx_closure_new_func_payload`'s ABI.
  pub(crate) type ClosureNewFn = unsafe extern "C" fn(
    fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure;

  /// Function-pointer type matching `mlx_closure_custom_new_func_payload`'s ABI.
  pub(crate) type ClosureCustomNewFn = unsafe extern "C" fn(
    fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure_custom;

  fn closure_new_slot() -> &'static Mutex<ClosureNewFn> {
    static SLOT: OnceLock<Mutex<ClosureNewFn>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(mlxrs_sys::mlx_closure_new_func_payload))
  }

  fn closure_custom_new_slot() -> &'static Mutex<ClosureCustomNewFn> {
    static SLOT: OnceLock<Mutex<ClosureCustomNewFn>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(mlxrs_sys::mlx_closure_custom_new_func_payload))
  }

  /// Read the currently-installed constructor (default: real mlx-c symbol).
  pub(crate) fn closure_new_fn() -> ClosureNewFn {
    *closure_new_slot()
      .lock()
      .expect("closure_new_slot poisoned")
  }

  /// Read the currently-installed custom-VJP constructor.
  pub(crate) fn closure_custom_new_fn() -> ClosureCustomNewFn {
    *closure_custom_new_slot()
      .lock()
      .expect("closure_custom_new_slot poisoned")
  }

  /// RAII guard: replace [`closure_new_fn`] with `stub` for the guard's
  /// lifetime, restore the real FFI symbol on drop. Acquires the seam
  /// mutex to serialize concurrent seam-test cases.
  pub(crate) struct ScopedClosureCtor {
    prev: ClosureNewFn,
  }

  impl ScopedClosureCtor {
    pub(crate) fn install(stub: ClosureNewFn) -> Self {
      let mut slot = closure_new_slot()
        .lock()
        .expect("closure_new_slot poisoned");
      let prev = *slot;
      *slot = stub;
      Self { prev }
    }
  }

  impl Drop for ScopedClosureCtor {
    fn drop(&mut self) {
      // Restore previous (real-FFI) symbol even if the test panicked.
      let mut slot = closure_new_slot()
        .lock()
        .expect("closure_new_slot poisoned");
      *slot = self.prev;
    }
  }

  /// Mirror of [`ScopedClosureCtor`] for the custom-VJP constructor seam.
  pub(crate) struct ScopedCustomCtor {
    prev: ClosureCustomNewFn,
  }

  impl ScopedCustomCtor {
    pub(crate) fn install(stub: ClosureCustomNewFn) -> Self {
      let mut slot = closure_custom_new_slot()
        .lock()
        .expect("closure_custom_new_slot poisoned");
      let prev = *slot;
      *slot = stub;
      Self { prev }
    }
  }

  impl Drop for ScopedCustomCtor {
    fn drop(&mut self) {
      let mut slot = closure_custom_new_slot()
        .lock()
        .expect("closure_custom_new_slot poisoned");
      *slot = self.prev;
    }
  }
}

#[cfg(test)]
mod tests {
  //! Deterministic regression tests for F1 (NULL-ctx UAF) via the
  //! [`test_seam`] function-pointer indirection.
  //!
  //! Pre-fix `Closure::new` and `closure_custom_new` reclaimed the payload
  //! via `Box::from_raw(payload_ptr.cast())` when mlx-c returned a NULL
  //! `ctx`. Per `mlx-c/mlx/c/closure.cpp` lines 70 / 471 mlx-c constructs a
  //! `std::shared_ptr<void>(payload, dtor)` as the first statement of the
  //! `try` block, so on any later throw the shared_ptr destructor has
  //! already invoked the registered Rust destructor (`destroy_payload` /
  //! `destroy_payload_3`) during stack unwinding — the Rust-side reclaim
  //! was a double-free / UAF.
  //!
  //! We can't deterministically inject OOM into mlx-c, so the integration
  //! tests in `tests/transforms.rs` only ever exercise the success path
  //! where `inner.ctx` is non-null — meaning a regression that
  //! re-introduced the reclaim would not surface in CI. These tests close
  //! that gap by swapping in a stub constructor that simulates the
  //! shared_ptr-then-throw failure mode: it invokes the registered
  //! destructor (proving mlx-c-side ownership transfer happened) and
  //! returns NULL `ctx`. A destructor-invocation counter then asserts that
  //! Rust did NOT also reclaim the box (count stays at 1, not 2).

  use std::sync::atomic::{AtomicUsize, Ordering};

  use super::{
    test_seam::{ClosureCustomNewFn, ClosureNewFn, ScopedClosureCtor, ScopedCustomCtor},
    *,
  };

  // ────────────── Closure::new NULL-ctx regression test ──────────────

  /// Per-test destructor-invocation counter for `destroy_payload`. The
  /// stub reads + bumps it; the test asserts the final count.
  static CLOSURE_DTOR_CALLS: AtomicUsize = AtomicUsize::new(0);

  /// Stub that simulates mlx-c's NULL-after-throw path for
  /// `mlx_closure_new_func_payload`:
  ///   1. Invoke the registered destructor on `payload` (mirroring the
  ///      `shared_ptr<void>(payload, dtor)` destructor that runs during
  ///      stack unwinding when the C++ ctor throws).
  ///   2. Return an `mlx_closure` with NULL `ctx` (mirroring the value the
  ///      `catch` clause returns to Rust).
  ///
  /// If `Closure::new` reclaims the payload via `Box::from_raw` on the
  /// NULL-ctx branch (the pre-fix bug), that's a second drop on the same
  /// pointer → ASAN double-free / UB. Post-fix it must NOT reclaim, so the
  /// destructor count after the call stays at exactly 1.
  unsafe extern "C" fn stub_closure_new_invokes_dtor_then_returns_null(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure {
    CLOSURE_DTOR_CALLS.fetch_add(1, Ordering::SeqCst);
    if let Some(d) = dtor {
      // SAFETY: `d` is `destroy_payload` (our own `extern "C"` fn), called
      // exactly once on the `payload` mlx-c received — same contract as
      // the real mlx-c implementation when its `try` block throws after
      // shared_ptr construction.
      unsafe { d(payload) };
    }
    mlxrs_sys::mlx_closure {
      ctx: ptr::null_mut(),
    }
  }

  /// Stub that returns NULL `ctx` WITHOUT invoking the destructor: models
  /// the "alternate path" referenced by the SAFETY comment in
  /// `Closure::new` where mlx-c surfaces a NULL closure without ever
  /// constructing the shared_ptr. Per the documented contract the Rust
  /// wrapper accepts a tiny leak here over an undefined-behavior reclaim.
  /// The test asserts NO destructor call AND no UB.
  unsafe extern "C" fn stub_closure_new_returns_null_no_dtor(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    _payload: *mut c_void,
    _dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure {
    mlxrs_sys::mlx_closure {
      ctx: ptr::null_mut(),
    }
  }

  #[test]
  fn closure_new_returns_err_without_double_free_when_ffi_returns_null_after_invoking_destructor() {
    CLOSURE_DTOR_CALLS.store(0, Ordering::SeqCst);
    let _guard =
      ScopedClosureCtor::install(stub_closure_new_invokes_dtor_then_returns_null as ClosureNewFn);

    // User closure body never runs (stub returns NULL before trampoline
    // dispatch); we only need a `Fn(&[Array]) -> Result<Vec<Array>>` to
    // satisfy the type bound. `Array` is `!Clone`, so return an empty Vec
    // instead of cloning the input.
    let result = Closure::new(|_xs: &[Array]| Ok(Vec::<Array>::new()));

    assert!(
      result.is_err(),
      "Closure::new must surface Err when mlx-c returns NULL ctx"
    );

    // CRITICAL F1 regression assert: the stub invoked the destructor
    // exactly ONCE; the production code must NOT have reclaimed the box
    // a second time. Pre-fix this would have been 2 (double-free) and
    // ASAN/Miri would also trigger; in release it's a silent UAF.
    assert_eq!(
      CLOSURE_DTOR_CALLS.load(Ordering::SeqCst),
      1,
      "F1 REGRESSION: pre-fix Box::from_raw on the NULL-ctx branch would \
       have produced count=2 (double-free / UAF). Post-fix the destructor \
       runs exactly once (stub-invoked); Rust must not reclaim."
    );
  }

  #[test]
  fn closure_new_returns_err_without_uaf_when_ffi_returns_null_without_invoking_destructor() {
    // Reset shared counter (other test ran first or not — doesn't matter
    // because this stub doesn't bump it; we just assert it stayed zero
    // for THIS invocation by reading delta).
    let baseline = CLOSURE_DTOR_CALLS.load(Ordering::SeqCst);
    let _guard = ScopedClosureCtor::install(stub_closure_new_returns_null_no_dtor as ClosureNewFn);

    // Same reasoning as above re: `Array: !Clone` and stub-never-dispatches.
    let result = Closure::new(|_xs: &[Array]| Ok(Vec::<Array>::new()));

    assert!(
      result.is_err(),
      "Closure::new must surface Err when mlx-c returns NULL ctx (no-dtor path)"
    );
    // No destructor calls in this path: the leak-over-UAF contract.
    assert_eq!(
      CLOSURE_DTOR_CALLS.load(Ordering::SeqCst),
      baseline,
      "stub did not invoke destructor; counter must not advance — and \
       crucially Rust must not call Box::from_raw on a pointer mlx-c \
       still owns (would be UAF on later mlx-c shared_ptr drop)."
    );
  }

  // ─────── closure_custom_new NULL-ctx regression test ───────

  /// Same counter discipline as [`CLOSURE_DTOR_CALLS`], for the
  /// `BoxedFn3` payload (`destroy_payload_3`).
  static CUSTOM_DTOR_CALLS: AtomicUsize = AtomicUsize::new(0);

  unsafe extern "C" fn stub_closure_custom_new_invokes_dtor_then_returns_null(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure_custom {
    CUSTOM_DTOR_CALLS.fetch_add(1, Ordering::SeqCst);
    if let Some(d) = dtor {
      // SAFETY: `d` is `destroy_payload_3`, invoked exactly once on the
      // `payload` mlx-c received.
      unsafe { d(payload) };
    }
    mlxrs_sys::mlx_closure_custom {
      ctx: ptr::null_mut(),
    }
  }

  #[test]
  fn closure_custom_new_returns_err_without_double_free_when_ffi_returns_null_after_invoking_destructor()
   {
    CUSTOM_DTOR_CALLS.store(0, Ordering::SeqCst);
    let _guard = ScopedCustomCtor::install(
      stub_closure_custom_new_invokes_dtor_then_returns_null as ClosureCustomNewFn,
    );

    let result = closure_custom_new(|_p: &[Array], _o: &[Array], _c: &[Array]| Ok(Vec::new()));

    assert!(
      result.is_err(),
      "closure_custom_new must surface Err when mlx-c returns NULL ctx"
    );

    // F1 regression assert for the BoxedFn3 path.
    assert_eq!(
      CUSTOM_DTOR_CALLS.load(Ordering::SeqCst),
      1,
      "F1 REGRESSION (custom-VJP): pre-fix Box::from_raw on the NULL-ctx \
       branch would have produced count=2 (double-free / UAF). Post-fix \
       the destructor runs exactly once (stub-invoked); Rust must not reclaim."
    );
  }
}
