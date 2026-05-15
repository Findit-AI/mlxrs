//! Goal 7 perf-floor sanity check. Single-thread / single-stream tripwire.
//!
//! This is **not** a benchmark — it just catches an order-of-magnitude
//! regression in the canonical add-loop. On Apple silicon (M2-class) the loop
//! typically runs in single-digit milliseconds; the threshold is set
//! deliberately loose so noisy CI hardware doesn't flake. If you're seeing
//! this fire on a real PR, the culprit is usually an FFI-per-op regression
//! (e.g. accidental clone in the wrapper) — not a Metal-side change.
//!
//! Run release-mode: `cargo test -p mlxrs --release --test perf_floor`.

use std::time::Instant;

use mlxrs::{Array, ops};

#[test]
fn perf_floor_canonical_sequence_under_500ms() {
  let mut a = Array::ones::<f32>(&(1024usize, 1024)).expect("ones");
  let b = Array::ones::<f32>(&(1024usize, 1024)).expect("ones");
  let start = Instant::now();
  for _ in 0..100 {
    // UFCS free-fn form avoids the move-then-call borrow-checker conflict.
    a = ops::arithmetic::add(&a, &b).expect("add");
  }
  a.eval().expect("eval");
  let elapsed = start.elapsed();
  // Loose tripwire: the CI matrix runs in debug mode for some entries (where
  // 100x 1024² adds + lazy-graph build is slower) and on shared macos-14
  // runners that throttle. Debug + cold cache ~150ms; release + warm ~5ms.
  // 500ms catches >3× regression while staying robust to noise.
  assert!(
    elapsed.as_millis() < 500,
    "perf-floor exceeded: {} ms (>= 500 ms tripwire)",
    elapsed.as_millis()
  );
}
