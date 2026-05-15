//! `SharedArray` — `Send + Sync` wrapper for cross-thread `Array` access.
//!
//! `Array` is intentionally `!Send + !Sync` in M1 (see `array/mod.rs` doc
//! comment): cheap `Clone` is refcount-shared, so two clones across threads
//! would race on the underlying C++ `array_desc->status` mutated through
//! `const`. `SharedArray` fills the gap by wrapping `Arc<Mutex<Array>>`:
//!
//! - The `Mutex` serializes every access to the inner `Array`, so the
//!   const-mutation race is gone — only one thread holds an `&mut Array`
//!   at a time.
//! - `Arc` provides ergonomic, cheap cross-thread cloning of the *handle*
//!   to the same shared `Array`.
//! - `Arc<Mutex<T>>` is `Send + Sync` whenever `T: Send`, which is exactly
//!   what serialized access gives us — even though `Array` itself is
//!   `!Send`, exclusive locked access through the mutex is sound. We
//!   therefore add a manual `unsafe impl Send + Sync` on the newtype
//!   (the auto-derived bound would otherwise pick up `Array: !Send` and
//!   refuse to implement either trait).
//!
//! ## Mutex semantics
//!
//! - [`SharedArray::lock`] returns a guard whose `Deref` / `DerefMut` yields
//!   `&Array` / `&mut Array`. Mirrors `std::sync::Mutex::lock` ergonomics.
//! - Poisoning is **not recovered**. If a previous lock holder panicked
//!   while holding the guard, the underlying mlx state may be observably
//!   inconsistent (e.g. mid-eval). `lock` maps `PoisonError` to
//!   [`Error::Backend`] rather than offering `into_inner` recovery: callers
//!   that want a fresh state should drop the `SharedArray` and rebuild.
//! - [`SharedArray::try_lock`] returns `None` on contention *or* poisoning —
//!   for fast-path code that wants to skip rather than block; if you need to
//!   distinguish the two, use [`SharedArray::lock`].
//!
//! ## Drop
//!
//! `SharedArray` has no explicit `Drop` impl — the `Arc<Mutex<Array>>`
//! handles refcount + the inner `Array`'s `Drop` correctly when the last
//! `Arc` clone is dropped. The inner `Array::drop` already guarantees no
//! TLS / panic across `extern "C"`.
//!
//! ## Cross-thread `eval` caveat
//!
//! The Rust-level race is fully closed by the mutex, but mlx itself stores
//! the default GPU stream and per-stream command encoders in C++ TLS (see
//! `stream.rs` docs). A `SharedArray` constructed on thread A and `eval`'d
//! on thread B fails with `"There is no Stream(gpu, N) in current thread."`
//! Two safe patterns:
//!
//! - **Construct on the worker.** Build the `Array` (and so the
//!   `SharedArray`) inside the thread that will eval it, then share the
//!   `SharedArray` back out for read-only access elsewhere.
//! - **Read-only access from anywhere.** `shape()`, `dtype()`, `ndim()`,
//!   `size()` don't touch a stream, so any thread that holds the `lock`
//!   guard can call them.
//!
//! A general "construct here, eval there" story needs the explicit `Stream`
//! API landing later in M2, which lets callers attach an array to a
//! particular stream and migrate it across threads with a documented
//! handshake.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{array::Array, error::Error};

/// `Send + Sync` newtype wrapping an `Arc<Mutex<Array>>` for cross-thread use.
///
/// Cloning is a cheap `Arc` refcount bump (no array data copy and no
/// `mlx_array_set` call — the inner `Array` is shared, not duplicated).
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, SharedArray};
///
/// let shared = SharedArray::new(Array::ones::<f32>(&(2, 2))?);
/// let s2 = shared.clone();
///
/// std::thread::spawn(move || -> mlxrs::Result<()> {
///   let mut g = s2.lock()?;
///   g.eval()?;
///   Ok(())
/// })
/// .join()
/// .unwrap()?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct SharedArray(Arc<Mutex<Array>>);

// SAFETY: `Array` is `!Send + !Sync` only because cheap `Clone` could let two
// threads observe `&mut Array` against the same underlying `array_desc`. The
// `Mutex` here serializes every access, so at most one `&mut Array` is alive
// across all threads at any time — the same invariant the compiler enforces
// for `Arc<Mutex<T>>` whenever `T: Send`. We assert `Send + Sync` manually
// because the auto-derived bound on `Arc<Mutex<T>>` requires `T: Send`, which
// `Array` doesn't impl.
unsafe impl Send for SharedArray {}
unsafe impl Sync for SharedArray {}

// Compile-time guarantees colocated with the type definition.
static_assertions::assert_impl_all!(SharedArray: Send, Sync, Clone);

impl SharedArray {
  /// Wrap an `Array` for cross-thread sharing.
  //
  // Clippy's `arc_with_non_send_sync` fires because `Mutex<Array>` looks
  // `!Send + !Sync` to the lint (it can't see our `unsafe impl` on the
  // newtype). The `Mutex<Array>` is in fact only ever accessed through a
  // `SharedArray`, where the manual `Send + Sync` impl makes it safe; suppress
  // the lint locally rather than globally.
  #[allow(clippy::arc_with_non_send_sync)]
  pub fn new(arr: Array) -> Self {
    Self(Arc::new(Mutex::new(arr)))
  }

  /// Lock the shared `Array` for exclusive use. Blocks the current thread
  /// until the mutex is acquired. The returned guard derefs to `&mut Array`.
  ///
  /// Returns [`Error::Backend`] if the mutex was poisoned by a prior panic;
  /// see the module docs for the no-recovery rationale.
  pub fn lock(&self) -> crate::error::Result<MutexGuard<'_, Array>> {
    self.0.lock().map_err(|_| Error::Backend {
      message: "SharedArray mutex poisoned (a previous lock holder panicked)".into(),
    })
  }

  /// Non-blocking lock attempt. Returns `None` if the mutex is contended
  /// **or** poisoned — callers needing to distinguish the two cases should
  /// use [`SharedArray::lock`].
  pub fn try_lock(&self) -> Option<MutexGuard<'_, Array>> {
    self.0.try_lock().ok()
  }

  /// Consume the `SharedArray` and return the inner `Array` if this is the
  /// last live handle. Returns `None` if other clones still hold the `Arc`
  /// — the caller can re-wrap with [`SharedArray::new`] after dropping
  /// outstanding clones, or use [`SharedArray::lock`] to operate in place.
  ///
  /// Returns `None` (rather than `Err`) on poisoning to match the
  /// `Arc::try_unwrap` shape: the caller already had to handle the
  /// "still aliased" fallback path, so collapsing both into `None` keeps
  /// the API surface narrow. The mlx state may be inconsistent in that
  /// case anyway (see module docs).
  pub fn into_inner(self) -> Option<Array> {
    Arc::try_unwrap(self.0)
      .ok()
      .and_then(|m| m.into_inner().ok())
  }
}
