//! Neural-network primitives ported from `mlx.nn`
//! (python `python/mlx/nn/layers/`) and the mlx-swift `MLXNN` / `MLXLMCommon`
//! layers, scoped to what the `lm` inference stack composes.
//!
//! M-N1 lands the base **Rotary Position Embedding**
//! ([`mod@rope`]) — the standard / "traditional" RoPE that backs every
//! attention layer's positional encoding (mlx-lm's `nn.RoPE`, swift's
//! `RoPE` + `MLXFast.RoPE`).
//!
//! The scaled RoPE variants (Llama3 / Su-scaled (longrope) / YaRN / NTK
//! interpolation — swift `Llama3RoPE` / `SuScaledRoPE` / `YarnRoPE` in
//! `MLXLMCommon/RoPEUtils.swift`) are deliberately **out of scope here**:
//! they precompute a per-dimension `freqs` array and forward it through the
//! same `mlx_fast_rope` primitive with `base = None`. They will land as
//! sibling modules under `lm::nn`; adding them will extend the shared
//! `mlx_fast_rope` wrapper with a `freqs`-based entry point (today
//! [`rope::rope`] exposes only the `base` path).
//!
//! M-N4 adds the **normalization** primitives ([`mod@norm`]):
//! [`RMSNorm`] / [`LayerNorm`] (both wrapping the fused `mlx_fast_*`
//! kernels — same primitives mlx-lm's `nn.RMSNorm` / `nn.LayerNorm` and
//! swift `RMSNorm` / `LayerNorm` delegate to) and [`GroupNorm`]
//! (no fused kernel; reproduced via [`crate::ops`]). The
//! `BatchNorm` / `InstanceNorm` siblings are deferred — these three
//! cover ~all transformer LM/VLM use.

pub mod norm;
pub mod rope;

pub use norm::{GroupNorm, LayerNorm, RMSNorm};
pub use rope::{Rope, RopeOffsetRef, rope, rope_dynamic, rope_with_offset};
