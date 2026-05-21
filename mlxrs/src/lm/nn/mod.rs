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
//! sibling modules under `lm::nn` that reuse [`rope::rope`]'s freqs path,
//! so this module is structured to grow into them without churn.

pub mod rope;

pub use rope::{Rope, rope};
