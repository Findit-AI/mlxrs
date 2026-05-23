//! C6 — `pad_to_square` canvas fill micro-bench.
//!
//! Verify-before-claim (§5.4 of `docs/core-arch-simd-candidates.md` +
//! project memory **"Verify review premise empirically"**). Compares
//! the OLD per-3-byte `extend_from_slice` loop vs the NEW
//! `chunks_exact_mut(3) + copy_from_slice` scalar reference vs the
//! NEW NEON 48-byte LCM(3, 16) pre-broadcast tile at three canvas
//! sizes (256² / 1024² / 4096²).
//!
//! Decision rule: if NEON is < 2× faster than the new scalar at
//! 4096² the NEON kernel is dropped (LLVM already crushes the
//! pattern). Numbers + the implementation decision are reported in
//! the final commit message and in the C6 section of the local
//! `docs/core-arch-simd-candidates.md`.
//!
//! Cargo `[profile.bench]` already pins `opt-level = 3`,
//! `codegen-units = 1`, `lto = 'thin'` in the workspace root
//! `Cargo.toml` — no per-bench tuning needed.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::vlm::pad_canvas_fill::{pad_canvas_fill, pad_canvas_fill_scalar};

// `[1, 128, 254]` is the asymmetric triple used in the C6 differential
// tests — it makes any pattern-broadcast bug visible (a kernel that
// writes `[1, 1, 1]` or `[1, 128, 254, 128, 254, 1, ...]` is wrong).
// Same triple here so the kernel runs are byte-identical to the test
// path — eliminates any "constant-folded by the optimizer" risk.
const RGB: [u8; 3] = [1, 128, 254];

/// Pre-C6 idiom — `Vec::with_capacity` + per-3-byte `extend_from_slice`.
/// **Not** the dispatcher; this is the legacy code we replaced. The
/// `Vec::with_capacity` matches the `try_reserve_exact` cap discipline
/// at the call site (one allocation, no realloc inside the loop).
#[inline(never)]
fn old_extend_loop(bytes: usize) -> Vec<u8> {
  let mut buf: Vec<u8> = Vec::with_capacity(bytes);
  for _ in 0..(bytes / 3) {
    buf.extend_from_slice(&RGB);
  }
  buf
}

/// NEW scalar reference — `chunks_exact_mut(3) + copy_from_slice` on
/// a pre-sized `vec![0u8; bytes]`. Matches the dispatcher's scalar
/// arm bit-for-bit. The allocation is `vec![0u8; bytes]` rather than
/// the call-site's `try_reserve_exact + spare_capacity_mut + set_len`
/// dance so the bench measures the fill kernel — not the alloc
/// pattern. The `vec![0u8; n]` zero-fill is identical across all three
/// benches and effectively a `memset`, so any timing difference between
/// the three benches isolates the fill loop.
#[inline(never)]
fn new_scalar(bytes: usize) -> Vec<u8> {
  let mut buf = vec![0u8; bytes];
  pad_canvas_fill_scalar(&mut buf, RGB);
  buf
}

/// NEW dispatcher — on `aarch64` routes to the NEON 48-byte LCM(3,
/// 16) tile, elsewhere to `pad_canvas_fill_scalar`. Same allocation
/// shape as `new_scalar` so the timing diff isolates the NEON body.
#[inline(never)]
fn new_dispatch(bytes: usize) -> Vec<u8> {
  let mut buf = vec![0u8; bytes];
  pad_canvas_fill(&mut buf, RGB);
  buf
}

fn bench_pad_canvas_fill(c: &mut Criterion) {
  // 256² (≈196 kB), 1024² (≈3.1 MB), 4096² (≈50 MB) — three orders of
  // magnitude across L1 / L2 / DRAM working sets.
  for &side in &[256usize, 1024, 4096] {
    let bytes = side * side * 3;
    let mut group = c.benchmark_group(format!("pad_canvas_fill/{side}x{side}"));
    // Tell criterion the input-byte rate so it reports GB/s alongside
    // the wall-clock ns/iter. Memory bandwidth is the natural ceiling
    // for this kernel.
    group.throughput(criterion::Throughput::Bytes(bytes as u64));

    group.bench_function("old_extend_loop", |b| {
      b.iter(|| {
        let v = old_extend_loop(black_box(bytes));
        black_box(v);
      });
    });
    group.bench_function("new_scalar", |b| {
      b.iter(|| {
        let v = new_scalar(black_box(bytes));
        black_box(v);
      });
    });
    group.bench_function("new_dispatch", |b| {
      b.iter(|| {
        let v = new_dispatch(black_box(bytes));
        black_box(v);
      });
    });

    group.finish();
  }
}

criterion_group!(benches, bench_pad_canvas_fill);
criterion_main!(benches);
