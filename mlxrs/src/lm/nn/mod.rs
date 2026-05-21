//! Neural-network primitives ported from `mlx.nn`
//! (python `python/mlx/nn/layers/`) and the mlx-swift `MLXNN` / `MLXLMCommon`
//! layers, scoped to what the `lm` inference stack composes.
//!
//! M-N1 lands the base **Rotary Position Embedding**
//! ([`mod@rope`]) — the standard / "traditional" RoPE that backs every
//! attention layer's positional encoding (mlx-lm's `nn.RoPE`, swift's
//! `RoPE` + `MLXFast.RoPE`).
//!
//! M-N2 adds the fast scaled-dot-product **attention** primitive
//! ([`mod@attention`]) — a 1:1 wrap of mlx's
//! `mx.fast.scaled_dot_product_attention` /
//! `MLXFast.scaledDotProductAttention` (`mlx_fast_scaled_dot_product_attention`),
//! covering Multi-Head, Grouped Query, and Multi-Query attention with
//! `None` / `Causal` / explicit-array masks.
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
//! The cache-aware quantized routing variant of attention
//! (swift `attentionWithCacheUpdate`'s `QuantizedKVCacheProtocol` branch
//! dispatching to `quantizedScaledDotProductAttention`) and the attention
//! `sinks` argument are likewise deliberately out of scope here — both are
//! follow-ups layered on top of the base [`attention::scaled_dot_product_attention`].

pub mod attention;
pub mod rope;

pub use attention::{Mask, scaled_dot_product_attention};
pub use rope::{Rope, RopeOffsetRef, rope, rope_dynamic, rope_with_offset};
