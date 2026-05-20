//! Vision-Language Model (VLM) support — multimodal inference building
//! blocks ported from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) and the
//! Swift [`MLXVLM`](https://github.com/ml-explore/mlx-swift-examples)
//! library.
//!
//! M4 ships the model-agnostic prompt-assembly primitives
//! ([`crate::vlm::prompt`]) — image-token splicing, image-span location, and
//! multimodal attention-mask construction — that downstream VLM forward
//! passes consume. Per-model architectures (Qwen-VL/LLaVA/etc.) are added
//! per-usecase, not bulk-ported from `mlx-vlm/models/`.

pub mod prompt;
