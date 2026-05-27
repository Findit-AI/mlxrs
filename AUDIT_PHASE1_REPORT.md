# mlxrs Adversarial Audit Report — Phase 1: FFI Foundation Layer

**Date:** 2026-05-27
**Auditor:** 6 Expert Teams (FFI Safety, Numerical Correctness, API Design,
           Concurrency, SIMD/Performance, Adversarial/Red Team)
**Scope:** R1-R8 — mlxrs-sys, device, dtype, shape, error, io, lib.rs
**Files:** 8 files, ~1,700 lines reviewed by all 6 teams independently

---

## Cross-Team Finding Summary (deduplicated)

| # | Severity | Category | Finding | Teams |
|---|----------|----------|---------|-------|
| 1 | HIGH | API | Dtype missing `FromStr` impl — users parsing configs must build own match | API |
| 2 | HIGH | API | Shape.rs "zero-allocation" claim not qualified for rank > 8 (falls back to Vec) | API |
| 3 | HIGH | API | Error enum size (~144B) not documented/guarded with static_assert | API |
| 4 | HIGH | API | Device and Dtype missing `Hash` derive (Copy types expected to be Hash) | API, Concurrency |
| 5 | MEDIUM | FFI | Device::Debug leaks `mlx_string` handle on panic (missing RAII guard) | FFI |
| 6 | MEDIUM | Numerical | Complex64 has no safe data extraction path (no Element impl, undocumented gap) | Numerical, API |
| 7 | MEDIUM | API | Device missing `Display` impl (only has Debug via FFI call) | API |
| 8 | MEDIUM | API | Error deprecated variants (`ShapeMismatch(String)`, `Backend(String)`) lack `#[deprecated]` attribute | API |
| 9 | MEDIUM | API | Shape tuple impls only up to rank 4, not documented | API |
| 10 | MEDIUM | API | Shape missing `Vec<T>` impls (users must pass `&vec[..]`) | API |
| 11 | MEDIUM | API | Cross-cutting: `Device::current()` vs `get_default_stream()` naming inconsistency | API |
| 12 | MEDIUM | API | Feature-gated Error variants change enum size — not documented | API |
| 13 | LOW | FFI | Device::Drop called on NULL-ctx handle (try_clone failure path) | FFI |
| 14 | LOW | FFI/Numerical | `shape[i] as usize` in conversion.rs:317 — negative c_int wraps to huge usize | FFI, Numerical |
| 15 | LOW | Numerical | Misleading comment on f16 transmute_copy rationale (says avoids Copy, but type IS Copy) | Numerical |
| 16 | LOW | Adversarial | Missing NULL check on `mlx_string_data` in Array::Display (conversion.rs:376) | Adversarial, FFI |
| 17 | LOW | API | Device::equal() public method duplicates PartialEq impl | API |
| 18 | LOW | API | Dtype #[non_exhaustive] decision — version coupling point, documented | API |
| 19 | LOW | API | lib.rs re-exports don't include error payload types | API |
| 20 | SUGGESTION | FFI | Build script silently skips submodule check when git not on PATH | FFI |
| 21 | SUGGESTION | API | Error enum missing top-level doc explaining "large Error" design rationale | API |
| 22 | SUGGESTION | API | `Device::with_index` negative index not validated (passed to mlx-c unchecked) | API |
| 23 | SUGGESTION | Numerical | `is_row_contiguous` saturating_mul could silently return false on overflow | Numerical |
| 24 | SUGGESTION | Numerical | `IntoShape` for `&[i32]` doesn't call validate_dims (caller responsibility) | Numerical |
| 25 | SUGGESTION | Concurrency | No runtime thread stress tests (compile-time !Send/!Sync tests present) | Concurrency |

---

## Team-by-Team Results

### Team 1: FFI Safety Expert
**Result:** 0 CRITICAL, 0 HIGH, 1 MEDIUM, 2 LOW, 1 SUGGESTION, 13 PASS

Key finding: Device::Debug creates an `mlx_string` handle without RAII protection.
The `write!` or `to_string_lossy()` could panic (OOM), leaking the handle.
Compare with Array::Display which correctly uses `StringGuard` RAII.

### Team 2: Numerical Correctness Expert
**Result:** 0 CRITICAL, 0 HIGH, 1 MEDIUM, 3 LOW, 3 SUGGESTION

Key finding: All 14 dtype mappings verified correct 1:1. `mlx_array_size` confirmed
as element count (not byte count). `transmute_copy` for f16/bf16 verified sound.
Complex64 has no Element impl — users can create complex arrays but can't read them back.

### Team 3: API Design Expert
**Result:** 0 CRITICAL, 4 HIGH, 8 MEDIUM, 6 LOW, 6 SUGGESTION

Key findings: Dtype missing `FromStr` (common ergonomics gap), Error enum size
not guarded with static_assert, Device/Dtype missing Hash derive, shape.rs
"zero-allocation" claim not qualified for rank > 8.

### Team 4: Concurrency/Thread Safety Expert
**Result:** 0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW, 0 SUGGESTION — FULL PASS

All global mutable state properly synchronized (15+ statics audited).
`DEFAULT_DEVICE_LOCK` correct. Poison handling correct. `!Send`/`!Sync`
compile-time enforcement via `static_assertions`. No bare `static mut` found.

### Team 5: SIMD/Performance Expert
**Result:** 0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW, 0 SUGGESTION — near-full PASS

Foundation layer well-engineered: zero-allocation shape conversion, borrowed-pointer
data access, zero-copy as_slice for contiguous arrays. NEON vs scalar bit-identical.

### Team 6: Adversarial/Red Team Expert
**Result:** 0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW, 0 SUGGESTION — well-defended

13 attack vectors tested. All failed against the safe API. The only finding is a
missing NULL check on `mlx_string_data` in Array::Display (defense-in-depth gap).
`into_raw`/`from_raw` correctly unsafe-gated. Drop is panic-free by design.

---

## PASS Categories (all 6 teams agree)

- repr(transparent) correctness
- unsafe impl Send/Sync soundness
- Element::sentinel_ptr() validity
- validate_dims completeness
- dim_ptr/stride_ptr empty-slice handling
- Handle lifecycle (create → use → free)
- FFI boundary return codes / null handles
- transmute_copy for f16/bf16
- Drop ordering and panic safety
- Thread safety — no data races
- Build script robustness
- Type confusion prevention
- SIMD dispatch correctness
- Scalar fallback quality

---

## Files Created

1. `/Users/joe/dev/mlxrs/AUDIT_PHASE1_REPORT.md` — This report

---

## Recommended Fixes (prioritized)

1. **[HIGH] Add `FromStr` to Dtype** — `impl FromStr for Dtype` mapping canonical strings
2. **[HIGH] Add `Hash` to Device and Dtype** — trivial derive addition
3. **[HIGH] Document Error enum size** — add `static_assert!(size_of::<Error>() <= 192)`
4. **[HIGH] Document shape.rs rank > 8 heap fallback**
5. **[MEDIUM] Fix Device::Debug RAII** — use StringGuard pattern from Array::Display
6. **[MEDIUM] Add NULL check in Array::Display** — match Device::Debug's pattern
7. **[MEDIUM] Add `#[deprecated]` to ShapeMismatch/Backend error variants**
8. **[MEDIUM] Document Complex64 limitation** — or add Element impl if FFI functions exist
