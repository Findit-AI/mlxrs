//! Stream module: internal per-thread default GPU stream + public `Stream`
//! handle (M2).
//!
//! ## Internal singleton (M1 carry-over)
//!
//! `default_stream()` is a per-thread cache of the default GPU stream used by
//! every `ops::*` free function. It is intentionally process-lifetime-leaked
//! (Metal frameworks tear down before destructors run, so calling
//! `mlx_stream_free` at exit would crash).
//!
//! Per-thread (not process-wide) because mlx-c++ stores the default stream and
//! its `CommandEncoder` in `thread_local` storage on the C++ side
//! (see `mlx/stream.cpp::default_stream_storage` and
//! `mlx/backend/metal/device.cpp::get_command_encoders`). A handle obtained on
//! one thread cannot be used to eval on another — eval throws
//! "There is no Stream(gpu, N) in current thread."
//!
//! ## Public `Stream` (M2)
//!
//! [`Stream`] is an explicit, owned RAII handle for callers who want lifetime
//! control: short-lived worker threads, multi-device pipelines, or fixtures
//! that need deterministic teardown. Drop calls `mlx_stream_free`. Same
//! per-thread caveat applies for GPU streams — a `Stream` obtained on thread
//! T can only be used to eval on T (mlx-c++ side asserts this; we cannot
//! enforce it at compile time without giving up `Send`, which the audit
//! confirms is sound for the POD `mlx::core::Stream`).

use std::{cell::Cell, ffi::CStr};

use static_assertions::assert_not_impl_any;

use crate::{
  device::Device,
  error::{Result, check, ensure_handler_installed},
};

thread_local! {
  static DEFAULT_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
}

pub(crate) fn default_stream() -> mlxrs_sys::mlx_stream {
  // Most safe-layer FFI consumers funnel through here; install the error
  // handler before any mlx-c call so a stripped/disabled #[ctor] cannot let
  // the default printf+exit handler fire on the very first failure.
  crate::error::ensure_handler_installed();
  DEFAULT_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: handler installed above; errors surface via TLS.
    let s = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if s.ctx.is_null() {
      panic!(
        "mlxrs: mlx_default_gpu_stream_new returned NULL ctx — \
         GPU unavailable or initialization failed. Aborting."
      );
    }
    cell.set(Some(s));
    s
  })
}

/// Invalidate this thread's cached default-stream handle so the next
/// [`default_stream`] call re-creates it via `mlx_default_gpu_stream_new`.
///
/// Required after `mlx::core::clear_streams()`: that destroys the thread's
/// Metal command encoders, so the cached `{gpu, 0}` handle is now stale
/// (eval against it fails with "There is no Stream(gpu, 0) in current
/// thread"). Resetting the cache lets the next op re-register a fresh
/// encoder. See [`super::Stream::clear_current_thread_streams`].
pub(crate) fn reset_default_stream_cache() {
  DEFAULT_STREAM.with(|cell| cell.set(None));
}

// INTENTIONAL: never freed at thread/process exit. Metal frameworks tear down
// before destructors run, so calling mlx_stream_free at exit would crash.
// Instruments will flag this as a leak on shutdown — that's expected.
//
// USAGE GUIDANCE: each thread that ever calls into mlxrs allocates its own
// GPU stream that lives until process exit. mlxrs is intended to be driven
// from a small, long-lived set of worker threads (a fixed-size thread pool
// or the main thread). Patterns that spawn a fresh OS thread per request or
// per task — rayon-with-thread-recycling, std::thread per HTTP request,
// short-lived spawn loops — will accumulate one mlx_stream per worker over
// the process lifetime and grow without bound. M2's public `Stream` API
// (below) provides explicit lifetime control for those cases.

// ───────────────────────── Public Stream API (M2) ─────────────────────────

/// MLX execution stream — RAII handle around `mlxrs_sys::mlx_stream`.
///
/// A stream targets a specific device and serializes work submitted to it.
/// Construct via [`Stream::default_gpu`], [`Stream::default_cpu`], or
/// [`Stream::new_on`]. Drop calls `mlx_stream_free`.
///
/// ## Threading
/// `Stream` is intentionally **`!Send` and `!Sync`**.
///
/// The `mlx::core::Stream` struct is a `{DeviceType, int}` POD, so the
/// Phase-3 audit originally concluded Send/Sync was sound. That conclusion
/// was layout-only and is wrong in practice: a `Stream` is an *index into
/// per-thread state*. mlx-c++ stores the default-stream and the per-stream
/// `CommandEncoder` in C++ thread-local storage, so a GPU stream constructed
/// on thread A cannot be used to eval (or `synchronize`) on thread B —
/// mlx-c++ throws `"There is no Stream(gpu, N) in current thread."`. This
/// was confirmed empirically by the `SharedArray` cross-thread experiment.
///
/// This is the same class of bug as the M1 `Array` Send revision: a
/// trivially-copyable handle whose *referent* has thread-affine state.
/// Marking the wrapper `Send` would let safe code move the handle across a
/// thread boundary and hit that failure path. Until a thread-checked or
/// CPU/GPU-split API exists (future milestone), `Stream` stays single-thread
/// like `Array`. (`Device` IS `Send + Sync` — it is a pure `{kind, index}`
/// descriptor with no thread-local referent.)
///
/// # Lifetime contract — NOT per-value RAII
///
/// `Stream` is a `Drop` type, but **`Drop` only frees the small C handle
/// box** (`delete (mlx::core::Stream*)ctx`) — it does NOT reclaim the
/// underlying mlx stream. mlx's stream model:
/// - `mlx::core::new_stream` appends `{index, device}` to a process-global
///   `std::vector<Stream>` (no removal API) and, for GPU, registers a Metal
///   command encoder in *thread-local* storage.
/// - mlx's ONLY teardown primitive is `mlx::core::clear_streams()`, which
///   is **thread-wide and bulk** ("destroy all streams created on the
///   current thread" — it clears that thread's command-encoder map). There
///   is no per-stream free, so this fundamentally cannot map to Rust
///   per-value `Drop`. mlx-c does not expose it either; mlxrs bridges it
///   via a first-party shim — see [`Stream::clear_current_thread_streams`].
///
/// Consequences:
/// - [`Stream::default_gpu`] / [`Stream::default_cpu`] are cheap — they
///   return the pre-existing per-thread default; no registry growth.
/// - [`Stream::new_on`] permanently grows the global registry (+ a GPU
///   command encoder) on every call. `Drop` does NOT give that back.
///   Create a bounded set once at startup, never per request/task.
/// - To bound encoder memory in a worker-pool design, have each worker call
///   [`Stream::clear_current_thread_streams`] as its LAST mlx action before
///   the worker thread finishes (end-of-thread cleanup — mlx does not
///   re-bootstrap a thread's GPU stream afterward, so it is not a mid-life
///   "reset").
///
/// In short: streams are coarse, mostly-process-lifetime resources. Treat
/// `Stream` as a handle, not a scoped RAII guard.
#[repr(transparent)]
pub struct Stream(pub(crate) mlxrs_sys::mlx_stream);

// NO `unsafe impl Send/Sync for Stream`. The raw `mlx_stream` contains a
// `*mut c_void`, so the auto-traits are already absent; the assertion below
// locks that in against an accidental future `unsafe impl`.
assert_not_impl_any!(Stream: Send, Sync);

impl Drop for Stream {
  fn drop(&mut self) {
    // SAFETY: must NOT touch TLS or panic (drop runs during thread teardown).
    // Discard rc silently — same convention as Array::drop.
    //
    // IMPORTANT: this frees ONLY the small C handle box (`delete
    // (mlx::core::Stream*)ctx`). It does NOT reclaim the underlying mlx
    // stream. mlx-c++ has no stream-teardown API: `mlx::core::new_stream`
    // appends to a process-global `std::vector<Stream>` (and, for GPU,
    // allocates a Metal command queue) that lives until process exit. See
    // the `Stream` type docs for the lifetime contract — this is NOT
    // resource-reclaiming RAII.
    unsafe {
      let _ = mlxrs_sys::mlx_stream_free(self.0);
    }
  }
}

impl Clone for Stream {
  /// Independent handle that wraps a fresh `mlx_stream` ctx pointing at the
  /// same underlying `mlx::core::Stream` payload (same `{kind, index}`).
  fn clone(&self) -> Self {
    self
      .try_clone()
      .expect("Stream::clone: mlx_stream_set failed")
  }
}

impl Stream {
  /// The per-thread default GPU stream. Wraps `mlx_default_gpu_stream_new`.
  /// Cheap and repeatable — returns the thread's existing default, so it
  /// does NOT grow mlx's global stream registry (unlike [`Stream::new_on`]).
  /// See the type-level "Lifetime contract" note: `Drop` frees only the C
  /// handle box.
  ///
  /// On a thread that never spun up Metal, this triggers GPU initialization;
  /// returns `Err(Backend { .. })` if the GPU is unavailable.
  pub fn default_gpu() -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: "mlx_default_gpu_stream_new returned NULL ctx \
                  (GPU unavailable or init failed)"
          .into(),
      });
    }
    Ok(Self(raw))
  }

  /// New default-CPU stream. Wraps `mlx_default_cpu_stream_new`.
  pub fn default_cpu() -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_default_cpu_stream_new() };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: "mlx_default_cpu_stream_new returned NULL ctx".into(),
      });
    }
    Ok(Self(raw))
  }

  /// New distinct stream targeting `device`, for op pipelining /
  /// concurrency. Wraps `mlx_stream_new_device`.
  ///
  /// **PERMANENT ALLOCATION — read before calling in a loop.** mlx-c++'s
  /// `new_stream` appends to a process-global `std::vector<Stream>` with no
  /// removal path, and for a GPU device it also allocates a Metal command
  /// queue that is never reclaimed. Dropping the returned `Stream` frees
  /// only the tiny C handle box — NOT the registry slot or the command
  /// queue. Every `new_on` call therefore costs process-lifetime memory
  /// (and a GPU queue). Create a *bounded* set of streams once at startup;
  /// never one per request/task. (`default_gpu`/`default_cpu` do not have
  /// this cost — they return the pre-existing per-thread default.)
  pub fn new_on(device: &Device) -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_stream_new_device(device.0) };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: "mlx_stream_new_device returned NULL ctx".into(),
      });
    }
    Ok(Self(raw))
  }

  /// Refcount-style clone via `mlx_stream_set`. Returns `Result` so callers
  /// can handle the rare allocation-failure path explicitly.
  pub fn try_clone(&self) -> Result<Self> {
    ensure_handler_installed();
    // `mlx_stream_new` returns an empty handle (NULL ctx) intended to be
    // populated by `mlx_stream_set`/`mlx_get_default_stream` — same
    // out-param convention as `mlx_array_new`. Wrap in `Self` first so RAII
    // covers the fallible set.
    let mut out = Self(unsafe { mlxrs_sys::mlx_stream_new() });
    check(unsafe { mlxrs_sys::mlx_stream_set(&mut out.0, self.0) })?;
    Ok(out)
  }

  /// Block until all work submitted to this stream is complete. Wraps
  /// `mlx_synchronize`.
  pub fn synchronize(&self) -> Result<()> {
    ensure_handler_installed();
    check(unsafe { mlxrs_sys::mlx_synchronize(self.0) })
  }

  /// Destroy **every** stream created on the *current thread*, reclaiming
  /// their Metal command encoders in bulk. This is mlx's only stream-
  /// teardown primitive (`mlx::core::clear_streams()`); mlx-c does not
  /// expose it, so this calls a first-party C++ shim
  /// ([`mlxrs_sys::mlxrs_shim_clear_streams`]).
  ///
  /// # This is END-OF-THREAD cleanup, not a mid-life "reset"
  ///
  /// mlx does NOT re-bootstrap a thread's GPU stream after `clear_streams()`
  /// — empirically, even a fresh `mlx_default_gpu_stream_new()` afterward
  /// still fails eval with "There is no Stream(gpu, 0) in current thread".
  /// So the contract is strictly: **call this once, as the last mlx action
  /// on a worker thread, right before that thread finishes.** Do NOT
  /// continue doing mlx work on the thread afterward.
  ///
  /// The intended pattern is a fixed worker pool where each worker, before
  /// being joined/recycled, calls this to release its GPU encoder memory
  /// deterministically instead of leaking it until process exit (the
  /// otherwise-unavoidable cost of dynamic [`Stream::new_on`] usage). It is
  /// an associated function (not `&self`) because the operation is
  /// thread-wide and bulk — it cannot be scoped to one `Stream`; every
  /// `Stream` previously obtained on this thread (including the per-thread
  /// default) is invalidated.
  ///
  /// Returns `Err(Backend)` if the underlying C++ call threw (not expected
  /// in practice — it clears an `unordered_map`).
  pub fn clear_current_thread_streams() -> Result<()> {
    ensure_handler_installed();
    let rc = unsafe { mlxrs_sys::mlxrs_shim_clear_streams() };
    // Defensive hygiene: clear_streams() invalidated this thread's command
    // encoders, so the internally-cached default-stream handle now dangles.
    // Drop the cache unconditionally so that IF some later code on this
    // thread calls an op, default_stream() at least doesn't hand back a
    // known-dead {gpu,0} handle (it will try a fresh new — which mlx itself
    // won't fully honor; that's why the doc says don't keep working on this
    // thread). This is not a "resume" guarantee, just not-eval-freed-state.
    reset_default_stream_cache();
    if rc == 0 {
      Ok(())
    } else {
      Err(crate::Error::Backend {
        message: "mlxrs_shim_clear_streams: mlx::core::clear_streams() threw".into(),
      })
    }
  }

  /// Returns the [`Device`] this stream targets. Wraps `mlx_stream_get_device`.
  pub fn device(&self) -> Result<Device> {
    ensure_handler_installed();
    let mut dev = Device(unsafe { mlxrs_sys::mlx_device_new() });
    check(unsafe { mlxrs_sys::mlx_stream_get_device(&mut dev.0, self.0) })?;
    Ok(dev)
  }

  /// Returns the index of this stream within its device. Wraps
  /// `mlx_stream_get_index`.
  pub fn index(&self) -> Result<i32> {
    ensure_handler_installed();
    let mut idx: i32 = 0;
    check(unsafe { mlxrs_sys::mlx_stream_get_index(&mut idx, self.0) })?;
    Ok(idx)
  }

  /// Whether two streams refer to the same `{device, index}` pair. Wraps
  /// `mlx_stream_equal`.
  pub fn equal(&self, other: &Stream) -> bool {
    unsafe { mlxrs_sys::mlx_stream_equal(self.0, other.0) }
  }

  /// Borrow the raw mlx-c handle (does not transfer ownership).
  ///
  /// # Safety
  /// Caller must not call `mlx_stream_free` on the returned handle and must
  /// not retain it past `self`'s lifetime.
  #[inline]
  pub unsafe fn as_raw(&self) -> mlxrs_sys::mlx_stream {
    self.0
  }
}

/// Returns the current process-wide default stream for `device`. Wraps
/// `mlx_get_default_stream`.
pub fn get_default_stream(device: &Device) -> Result<Stream> {
  ensure_handler_installed();
  let mut out = Stream(unsafe { mlxrs_sys::mlx_stream_new() });
  check(unsafe { mlxrs_sys::mlx_get_default_stream(&mut out.0, device.0) })?;
  Ok(out)
}

/// Install `stream` as the process-wide default for the device it targets.
/// Wraps `mlx_set_default_stream`.
///
/// Note: this does NOT swap the per-thread default-GPU stream cached by
/// `default_stream()` — internal `ops::*` calls keep using their cached
/// handle. Use this when interoperating with raw mlx-c calls or with the
/// `mlx_get_default_stream` API.
pub fn set_default_stream(stream: &Stream) -> Result<()> {
  ensure_handler_installed();
  check(unsafe { mlxrs_sys::mlx_set_default_stream(stream.0) })
}

impl PartialEq for Stream {
  fn eq(&self, other: &Self) -> bool {
    self.equal(other)
  }
}

impl Eq for Stream {}

impl std::fmt::Debug for Stream {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = unsafe { mlxrs_sys::mlx_string_new() };
    let rc = unsafe { mlxrs_sys::mlx_stream_tostring(&mut s, self.0) };
    let result = if rc == 0 {
      let p = unsafe { mlxrs_sys::mlx_string_data(s) };
      if p.is_null() {
        write!(f, "Stream(<unprintable>)")
      } else {
        let cs = unsafe { CStr::from_ptr(p) };
        write!(f, "Stream({})", cs.to_string_lossy())
      }
    } else {
      write!(f, "Stream(<unprintable>)")
    };
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(s);
    }
    result
  }
}
