//! Learning-rate schedules. Placeholder; full impl lands in commit 5.

#![allow(dead_code)]

/// Stub.
pub fn cosine_decay(_init: f32, _decay_steps: usize, _end: f32) -> Box<dyn Fn(usize) -> f32> {
  Box::new(|_| 0.0)
}
