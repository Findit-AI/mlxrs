//! `SharedArray` — cross-thread sharing wrapper around `Array`.

use mlxrs::{Array, SharedArray};

/// Runtime check that `SharedArray` is `Send + Sync + Clone`. The
/// `static_assertions::assert_impl_all!` macro inside the type's module
/// gives a compile-time error if these traits ever regress; this test
/// just ensures it stays linked into the test binary so the assertion
/// participates in `cargo test`.
#[test]
fn shared_array_send_sync_compile_assertions() {
  fn assert_send<T: Send>() {}
  fn assert_sync<T: Sync>() {}
  fn assert_clone<T: Clone>() {}
  assert_send::<SharedArray>();
  assert_sync::<SharedArray>();
  assert_clone::<SharedArray>();
}

#[test]
fn shared_array_lock_eval_works() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let shared = SharedArray::new(a);

  {
    let mut g = shared.lock().expect("lock");
    g.eval().expect("eval");
    let v = g.to_vec::<f32>().expect("to_vec");
    assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);
  }

  // Guard dropped — re-lock must succeed (mutex unlocked).
  let g2 = shared.lock().expect("relock after guard drop");
  assert_eq!(g2.shape(), vec![2, 2]);
}

#[test]
fn shared_array_try_lock_returns_some_when_uncontended() {
  let shared = SharedArray::new(Array::ones::<f32>(&(2,)).unwrap());
  let g = shared
    .try_lock()
    .expect("uncontended try_lock should succeed");
  assert_eq!(g.shape(), vec![2]);
}

#[test]
fn shared_array_try_lock_returns_none_when_held() {
  let shared = SharedArray::new(Array::ones::<f32>(&(2,)).unwrap());
  let _held = shared.lock().expect("lock");
  // Same thread, but std::sync::Mutex still reports WouldBlock when already locked.
  assert!(
    shared.try_lock().is_none(),
    "should fail to acquire while held"
  );
}

#[test]
fn shared_array_cross_thread_eval() {
  // mlx's GPU stream lives in C++ TLS (see `stream.rs` doc comment), so the
  // stream id baked into a freshly-constructed `Array` references the
  // **creating** thread's stream — and `eval` on a different thread fails
  // with "There is no Stream(gpu, N) in current thread."
  //
  // To test that the cross-thread `lock` machinery is sound under real eval,
  // we construct + eval on the same worker thread, then reuse the
  // `SharedArray` from the main thread for read-only inspection. (The
  // explicit `Stream` API landing later in M2 will let callers move arrays
  // across threads with a documented handshake; for now SharedArray's
  // cross-thread eval contract is "construct + eval on the same thread,
  // share for read-only access.")
  let shared: SharedArray = std::thread::spawn(|| -> SharedArray {
    let mut a = Array::ones::<f32>(&(2, 2)).unwrap();
    a.eval().unwrap();
    SharedArray::new(a)
  })
  .join()
  .expect("worker join");

  // Read-only access from the main thread: `to_vec` requires `eval` to have
  // completed, but the data buffer is shared across threads once materialized.
  let v: Vec<f32> = {
    let mut g = shared.lock().expect("lock from main");
    g.to_vec::<f32>().expect("to_vec")
  };
  assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);

  // And from a third thread — the clone moves into the spawned closure and
  // `shared` itself is unused after, so we let `shared` move directly.
  let v2 = std::thread::spawn(move || shared.lock().unwrap().to_vec::<f32>().unwrap())
    .join()
    .expect("third-thread join");
  assert_eq!(v2, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn shared_array_cross_thread_read_only_shape() {
  // Pure handle sharing without eval: `shape`/`dtype`/`ndim`/`size` don't
  // touch any thread-local stream, so this works regardless of the
  // originating thread. Demonstrates the "lightweight introspection"
  // cross-thread usecase.
  let shared = SharedArray::new(Array::ones::<f32>(&(3, 4)).unwrap());

  let (shape, ndim, size) = std::thread::spawn(move || {
    let g = shared.lock().unwrap();
    (g.shape(), g.ndim(), g.size())
  })
  .join()
  .expect("thread join");

  assert_eq!(shape, vec![3, 4]);
  assert_eq!(ndim, 2);
  assert_eq!(size, 12);
}

#[test]
fn shared_array_into_inner_happy_path() {
  let shared = SharedArray::new(Array::ones::<f32>(&(3,)).unwrap());
  let mut arr = shared
    .into_inner()
    .expect("sole owner — into_inner must succeed");
  assert_eq!(arr.shape(), vec![3]);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn shared_array_into_inner_returns_none_when_aliased() {
  let shared = SharedArray::new(Array::ones::<f32>(&(2,)).unwrap());
  let _alias = shared.clone();
  assert!(
    shared.into_inner().is_none(),
    "into_inner must return None while another Arc clone is alive"
  );
}
