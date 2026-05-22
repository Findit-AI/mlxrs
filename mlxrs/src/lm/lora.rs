//! Inference-time **LoRA / DoRA adapter loading** — the runtime surface that
//! takes a *pre-trained* low-rank adapter and runs it against a base model.
//!
//! Port of the inference-relevant half of mlx-lm's `mlx_lm/tuner/` and
//! mlx-swift-lm's `MLXLMCommon/Adapters/LoRA/`:
//!
//! - The LoRA-wrapped linear layer [`LoRALinear`] and the weight-decomposed
//!   [`DoRALinear`], each wrapping a [`BaseLinear`] that is **either** a dense
//!   weight **or** an MLX-quantized triple ([`BaseLinear::Quantized`]) — the
//!   QLoRA / QDoRA case (swift's separate `QLoRALinear` / `QDoRALinear`
//!   classes; here one type covers both bases, mirroring mlx-lm's `LoRALinear`
//!   which wraps `Linear` and `QuantizedLinear` alike, `tuner/lora.py:22-23`).
//!   These mirror mlx-lm `tuner/lora.py::LoRALinear` +
//!   `tuner/dora.py::DoRALinear` and the swift `LoRA+Layers.swift` /
//!   `DoRA+Layers.swift` classes. Each ports the FORWARD pass and the
//!   [`fuse`](LoraLayer::fuse) method (fold the adapter into the base weight so
//!   a fused model needs no adapter at runtime).
//! - [`linear_to_lora_layers`] — the layer-selection step: wrap the targeted
//!   linear weights of a [`Weights`] map (the `keys` / `num_layers` predicate
//!   from `adapter_config.json`), mirroring mlx-lm
//!   `tuner/utils.py::linear_to_lora_layers`.
//! - [`load_adapters`] — the load-time entry: read `adapter_config.json` +
//!   `adapters.safetensors` from a **local** directory (no HuggingFace Hub, per
//!   the project's local-path-only scope), build the LoRA/DoRA layers, and bind
//!   their parameters, mirroring mlx-lm `tuner/utils.py::load_adapters` +
//!   swift `LoRAContainer.from(directory:)` / `load(into:)`.
//! - [`LoraConfig`] — the `adapter_config.json` schema (rank `r`, `alpha` /
//!   `scale`, target `keys`, `fine_tune_type` lora|dora|full, `num_layers`),
//!   mirroring swift `LoRAConfiguration` (`LoRAContainer.swift:27-66`).
//!
//! # Scope — inference adapter loading ONLY
//!
//! This is the surface for **running** a pre-trained adapter, NOT training one.
//! Deliberately excluded (training, out of project scope): the optimizer /
//! loss / dataset / `trainer.py` surface (`tuner/trainer.py`,
//! `tuner/datasets.py`, `tuner/losses.py`), `print_trainable_parameters`, the
//! `dropout` (an inference adapter has no dropout — mlx-lm passes `dropout=0.0`
//! at load and the layer's `nn.Dropout(p=0)` is the identity; this port omits
//! the dropout module entirely, so [`LoRALinear::forward`] applies the
//! low-rank term to `x` directly), and the random `lora_a` / zero `lora_b`
//! *initializers* (training only — at inference both come from
//! `adapters.safetensors`).
//!
//! The MoE `LoRASwitchLinear` / `LoRAEmbedding` variants
//! (`tuner/lora.py:101,198`) are deferred follow-ups — they need the
//! `gather_mm`/embedding-as-linear adapter wiring layered on top of this base
//! `Linear` surface, exactly as [`crate::lm::nn::switch`] deferred `SwitchMLP`.
//!
//! # No module tree — the weight-map model
//!
//! mlx-lm / mlx-swift apply LoRA by walking a live `nn.Module` tree, replacing
//! `Linear` leaves with `LoRALinear` wrappers. mlxrs has **no** model-module
//! tree (that is per-usecase — [see project memory:
//! `feedback_no_per_model_arch_porting`]), so — exactly as
//! [`crate::lm::quant`] walks the [`Weights`] name-map instead of an
//! `nn.Module` — this module builds [`LoRALinear`] objects keyed by their
//! base-weight **path** in the loaded [`Weights`] map. [`linear_to_lora_layers`]
//! returns a [`LoraLayers`] map (path → wrapped layer); the per-usecase
//! architecture, which already routes a `model.layers.N.self_attn.q_proj` path
//! to its forward call, dispatches through the wrapped layer for adapted paths.
//!
//! # The LoRA forward math
//!
//! For a base linear `W` (shape `[output_dims, input_dims]`), low-rank factors
//! `lora_a` (`[input_dims, r]`) and `lora_b` (`[r, output_dims]`), and a scalar
//! `scale` (`scale = alpha / r` when `alpha`/`lora_alpha` is present — the PEFT
//! convention, which WINS over a literal `scale` — else the literal `scale`
//! field, else the `20.0` default):
//!
//! ```text
//! LoRA:  y = x @ Wᵀ (+ bias)
//!        z = (x @ lora_a) @ lora_b
//!        out = y + (scale · z)
//!
//! fuse:  Δ = (scale · lora_bᵀ) @ lora_aᵀ      // shape [output_dims, input_dims]
//!        W_fused = W + Δ
//! ```
//!
//! DoRA (weight-decomposed) additionally carries a per-output-row magnitude
//! `m = ‖W‖₂ along axis 1` (`[output_dims]`) and renormalizes:
//!
//! ```text
//! DoRA:  adapted = W + (scale · lora_bᵀ) @ lora_aᵀ
//!        denom   = ‖adapted‖₂ along axis 1
//!        out     = (m / denom) · (y + scale · z) (+ bias)
//!
//! fuse:  W_adapted = W + (scale · lora_bᵀ) @ lora_aᵀ
//!        W_fused   = (m / ‖W_adapted‖₂)[:, None] · W_adapted
//! ```
//!
//! These match mlx-lm `tuner/lora.py::LoRALinear.{__call__,fuse}` /
//! `tuner/dora.py::DoRALinear.{__call__,fuse}` and swift
//! `LoRA+Layers.swift` / `DoRA+Layers.swift` exactly.
//!
//! Conventions mirror [`crate::lm::quant`] / [`crate::lm::load`]:
//! `Result`-fallible, no implicit eval (the returned `Array`s are lazy — no
//! `eval`/`item`/`to_vec`), recoverable IO / parse / shape failures map to
//! [`Error::Backend`] / [`Error::ShapeMismatch`] with a clear message, and the
//! `adapter_config.json` read is bounded against an untrusted adapter directory
//! exactly as [`crate::lm::load::load_config`].
//!
//! [`Error::Backend`]: crate::Error::Backend
//! [`Error::ShapeMismatch`]: crate::Error::ShapeMismatch
//! [`feedback_no_per_model_arch_porting`]: crate::lm

use std::{
  collections::{HashMap, HashSet},
  path::Path,
};

use crate::{
  array::Array,
  error::{Error, Result},
  lm::{
    load::Weights,
    quant::{PerLayerQuantization, Quantization},
  },
  ops,
};

/// mlx-lm's default LoRA `scale` (`tuner/lora.py:17,73`: `scale: float =
/// 20.0`) and swift's (`LoRALinear` init `scale: Float = 20.0`). Applied when
/// neither `scale` nor `alpha` is present in `adapter_config.json`.
pub const DEFAULT_LORA_SCALE: f32 = 20.0;

/// mlx-lm's default LoRA rank (`tuner/lora.py:16,71`: `r: int = 8`).
pub const DEFAULT_LORA_RANK: i32 = 8;

/// mlx-lm's default `num_layers` for a LoRA config — the number of trailing
/// decoder blocks adapted (`mlx_lm` adapter configs commonly carry it
/// explicitly; swift `LoRAConfiguration` defaults to `16`,
/// `LoRAContainer.swift:52`).
pub const DEFAULT_NUM_LAYERS: i32 = 16;

/// Upper bound on the `adapters.safetensors` file [`load_adapters`] will hand
/// to [`crate::io::load_safetensors`]. A LoRA/DoRA adapter is **low-rank** —
/// only the `lora_a` / `lora_b` (and DoRA `m`) factors of the targeted
/// projections — so even a wide, high-rank adapter over a large model is well
/// under this bound; a file beyond it is not a plausible adapter. The cap
/// bounds the damage an untrusted adapter dir can do (a hostile
/// `adapters.safetensors` pointing at an oversized blob ⇒ a clear recoverable
/// error, not an OOM). Generous (2 GiB) because the budget is a safety ceiling,
/// not a tight fit — distinct from the 1-MiB `lm::load`-internal JSON-config
/// cap (`MAX_CONFIG_BYTES`).
pub const MAX_ADAPTER_SAFETENSORS_BYTES: u64 = 2 << 30;

// ───────────────────────────── config ─────────────────────────────

/// How a checkpoint was fine-tuned — mlx-lm `fine_tune_type`
/// (`tuner/utils.py:129`, one of `"lora"` / `"dora"` / `"full"`) and swift
/// `LoRAConfiguration.FineTuneType` (`LoRAContainer.swift:29-32`, `lora` /
/// `dora`).
///
/// `Full` (a full-weight fine-tune, no low-rank factorization) is recognized
/// for parity with mlx-lm `load_adapters` (which skips
/// `linear_to_lora_layers` entirely for `"full"` and just loads the dense
/// weight delta) but is **not** an adapter-wrapping mode — [`load_adapters`]
/// reports it as unsupported here, since mlxrs has no module tree to load a
/// full-weight delta into (the per-usecase architecture would merge a full
/// fine-tune at the weight-map level via [`crate::lm::load::load_weights`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FineTuneType {
  /// Low-Rank Adaptation — the `lora_a` / `lora_b` factors only
  /// (mlx-lm `LoRALinear`).
  Lora,
  /// Weight-Decomposed Low-Rank Adaptation — LoRA plus a learned per-row
  /// magnitude `m` (mlx-lm `DoRALinear`).
  Dora,
  /// Full-weight fine-tune (no low-rank factorization). Recognized but not an
  /// adapter-wrapping mode here — see the enum docs.
  Full,
}

impl Default for FineTuneType {
  /// `lora` — mlx-lm `getattr(config, "fine_tune_type", "lora")`
  /// (`tuner/utils.py:129`).
  fn default() -> Self {
    FineTuneType::Lora
  }
}

/// The `lora_parameters` sub-block of `adapter_config.json` — mlx-lm
/// `config.lora_parameters` (`tuner/utils.py:133`, a dict with `rank` /
/// `scale` / optional `keys` / `dropout` / `alpha`) and swift
/// `LoRAConfiguration.LoRAParameters` (`LoRAContainer.swift:34-45`).
///
/// `scale` is the literal low-rank scale (mlx-lm `config["scale"]`). When
/// `alpha` is present (the PEFT/HF convention `scale = alpha / rank`), it
/// **takes precedence** over the literal `scale`
/// ([`LoraParameters::resolved_scale`]). `keys` is the explicit
/// target-projection allowlist (e.g. `["self_attn.q_proj",
/// "self_attn.v_proj"]`); `None` means "every eligible linear" (mlx-lm's
/// auto-discovery, `tuner/utils.py:85-101`). `dropout` is carried for config
/// round-trip fidelity but **ignored at inference** (an inference adapter's
/// dropout is the identity — see the [module docs](self)).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LoraParameters {
  /// Low-rank dimension `r`. Deserializes from `rank` (mlx-lm
  /// `config["rank"]`) **or** the PEFT/HF key `r` (PEFT-trained adapters name
  /// it `r`, e.g. `"r": 16`) — mlx-lm-trained configs use `rank`, so accept
  /// both. Defaults to [`DEFAULT_LORA_RANK`] when neither is present.
  #[serde(default = "default_rank", alias = "r")]
  pub rank: i32,
  /// Literal low-rank scale (mlx-lm `config["scale"]`). Used when `alpha` is
  /// absent; defaults to [`DEFAULT_LORA_SCALE`] when neither `scale` nor
  /// `alpha` is present.
  #[serde(default)]
  pub scale: Option<f32>,
  /// PEFT/HF `lora_alpha` — if present, the effective scale is `alpha / rank`
  /// and this **takes precedence** over a literal `scale` (PEFT's `scaling =
  /// lora_alpha / r`). Carried so adapters trained with the HF convention load
  /// with the correct scale.
  #[serde(default, alias = "lora_alpha")]
  pub alpha: Option<f32>,
  /// Explicit target-projection allowlist (suffix paths like
  /// `"self_attn.q_proj"`). `None` ⇒ adapt every eligible linear.
  #[serde(default)]
  pub keys: Option<Vec<String>>,
  /// Training dropout — carried for round-trip fidelity, **ignored at
  /// inference** (see module docs).
  #[serde(default)]
  pub dropout: Option<f32>,
}

fn default_rank() -> i32 {
  DEFAULT_LORA_RANK
}

impl Default for LoraParameters {
  fn default() -> Self {
    Self {
      rank: DEFAULT_LORA_RANK,
      scale: None,
      alpha: None,
      keys: None,
      dropout: None,
    }
  }
}

impl LoraParameters {
  /// The effective low-rank scale, resolving the PEFT/HF precedence:
  /// `alpha` (`lora_alpha`) **wins** when present → `alpha / rank` (the HF
  /// convention an adapter trained with `lora_alpha` carries); else the literal
  /// `scale` field; else [`DEFAULT_LORA_SCALE`]. This matches PEFT's `scaling =
  /// lora_alpha / r` taking precedence over a stored scalar, and the
  /// [module docs](self) (`scale = alpha / r` when built from `alpha`, else the
  /// literal `scale`, else `20.0`).
  ///
  /// A non-positive `rank` with an `alpha` present cannot form `alpha / rank`,
  /// so it falls back to the literal `scale` (then the default) — the
  /// [`LoraConfig`]/[`load_adapters`] path rejects `rank <= 0` before a layer is
  /// ever built, so this is a defensive floor, not a live path.
  pub fn resolved_scale(&self) -> f32 {
    // `alpha` wins — but only when `rank > 0` can form `alpha / rank`. A
    // non-positive `rank` (or an absent `alpha`) falls through to the literal
    // `scale`, then the default.
    if let Some(a) = self.alpha
      && self.rank > 0
    {
      return a / self.rank as f32;
    }
    if let Some(s) = self.scale {
      s
    } else {
      DEFAULT_LORA_SCALE
    }
  }
}

/// The `adapter_config.json` schema — mlx-lm's adapter config
/// (`tuner/utils.py:127-136`: `fine_tune_type`, `num_layers`,
/// `lora_parameters`) and swift `LoRAConfiguration` (`LoRAContainer.swift:27-66`).
///
/// Forward-compatible by design (no `deny_unknown_fields`): an adapter config
/// carrying extra training-only keys (`optimizer`, `learning_rate`, `data`, …)
/// parses cleanly — exactly as mlx-lm reads it into a `SimpleNamespace` and
/// ignores the unused keys.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LoraConfig {
  /// `lora` / `dora` / `full` (mlx-lm `fine_tune_type`). Defaults to
  /// [`FineTuneType::Lora`] (mlx-lm's `getattr(..., "lora")`).
  #[serde(default)]
  pub fine_tune_type: FineTuneType,
  /// Number of trailing decoder blocks adapted (mlx-lm `config.num_layers`).
  /// Defaults to [`DEFAULT_NUM_LAYERS`]. A **non-positive** value selects ALL
  /// blocks, not none — mlx-lm's `model.layers[-max(num_layers, 0):]`
  /// (`tuner/utils.py:103`) reduces to `model.layers[-0:]` == `model.layers[0:]`
  /// when `num_layers <= 0` (the Python `-0` slice quirk), so `num_layers: -1`
  /// (and `0`) adapt every decoder block.
  #[serde(default = "default_num_layers")]
  pub num_layers: i32,
  /// The low-rank parameters block (mlx-lm `config.lora_parameters`).
  #[serde(default)]
  pub lora_parameters: LoraParameters,
  /// PEFT/HF `use_dora` flag — some adapters carry the DoRA signal here
  /// instead of `fine_tune_type: "dora"`. Either signal selects DoRA (see
  /// [`LoraConfig::is_dora`]).
  #[serde(default)]
  pub use_dora: bool,
}

fn default_num_layers() -> i32 {
  DEFAULT_NUM_LAYERS
}

impl LoraConfig {
  /// Parse a [`LoraConfig`] from an in-memory `adapter_config.json` string.
  ///
  /// Mirrors mlx-lm `json.load(adapter_config.json)` restricted to the typed
  /// subset. A serde failure (malformed JSON) maps to [`Error::Backend`] with
  /// the underlying cause — the codebase's config-parse error convention (twin
  /// of [`crate::lm::load::Config::from_json`]).
  pub fn from_json(json: &str) -> Result<LoraConfig> {
    serde_json::from_str(json).map_err(|e| Error::Backend {
      message: format!("invalid adapter_config.json: {e}"),
    })
  }

  /// Whether this config selects DoRA (weight-decomposed) — either
  /// `fine_tune_type: "dora"` or the PEFT `use_dora: true` flag, mirroring
  /// mlx-lm `use_dora=(fine_tune_type == "dora")` (`tuner/utils.py:135`) plus
  /// the HF `use_dora` convention.
  pub fn is_dora(&self) -> bool {
    self.fine_tune_type == FineTuneType::Dora || self.use_dora
  }

  /// The resolved low-rank scale (see [`LoraParameters::resolved_scale`]).
  pub fn scale(&self) -> f32 {
    self.lora_parameters.resolved_scale()
  }

  /// The low-rank dimension `r`.
  pub fn rank(&self) -> i32 {
    self.lora_parameters.rank
  }
}

// ───────────────────────── adapter weights ─────────────────────────

/// The per-layer adapter parameters loaded from `adapters.safetensors` for one
/// target path: the low-rank factors plus (DoRA only) the magnitude.
///
/// These are the *named* arrays mlx-lm's `LoRALinear` registers (`lora_a` /
/// `lora_b`) plus DoRA's `m` (`tuner/dora.py:90`). At inference they come
/// entirely from the safetensors file — there is no random/zero init here.
///
/// Does **not** derive `Clone` ([`Array`] deliberately doesn't — see
/// [`Array::try_clone`]); use [`AdapterParams::try_clone`] for the
/// refcount-sharing dup.
#[derive(Debug)]
pub struct AdapterParams {
  /// `lora_a` — shape `[input_dims, r]` (mlx-lm `tuner/lora.py:88-92`).
  pub lora_a: Array,
  /// `lora_b` — shape `[r, output_dims]` (mlx-lm `tuner/lora.py:93`).
  pub lora_b: Array,
  /// DoRA magnitude `m` — shape `[output_dims]` (mlx-lm `tuner/dora.py:90`);
  /// `None` for plain LoRA.
  pub magnitude: Option<Array>,
}

impl AdapterParams {
  /// Refcount-sharing dup of all three slots (a fresh mlx handle over the same
  /// buffer per [`Array::try_clone`]; no data copy). Fallible because the
  /// mlx-c handle alloc can fail.
  pub fn try_clone(&self) -> Result<Self> {
    Ok(Self {
      lora_a: self.lora_a.try_clone()?,
      lora_b: self.lora_b.try_clone()?,
      magnitude: match &self.magnitude {
        Some(m) => Some(m.try_clone()?),
        None => None,
      },
    })
  }
}

// ──────────────────────────── base layer ────────────────────────────

/// The base linear a LoRA/DoRA layer wraps: either a **dense** weight (+
/// optional bias) or an MLX-**quantized** triple. Mirrors the
/// `Linear` / `QuantizedLinear` split mlx-lm's `LoRALinear.from_base` branches
/// on (`tuner/lora.py:22-23`) and swift's `LoRALinear` / `QLoRALinear`.
///
/// Constructed via [`BaseLinear::dense`] / [`BaseLinear::quantized`] (the
/// quantized ctor validates the `affine`/`fp`-mode bias arity, mirroring
/// [`crate::lm::nn::switch::QuantizedSwitchLinear::from_parts`]); the inner
/// arrays are read-only thereafter so the `(weight, scales, biases)` triple
/// stays internally consistent.
#[derive(Debug)]
pub enum BaseLinear {
  /// Dense base: `weight` is `[output_dims, input_dims]`, `bias` optional
  /// `[output_dims]`.
  Dense {
    /// `[output_dims, input_dims]` dense weight.
    weight: Array,
    /// Optional `[output_dims]` bias (`None` ⇒ `bias=False`).
    bias: Option<Array>,
  },
  /// Quantized base: the MLX `(weight, scales, biases)` packed triple plus the
  /// scheme parameters, and an optional dense `[output_dims]` output bias.
  Quantized {
    /// Packed `uint32` quantized weight.
    weight: Array,
    /// Per-group scales.
    scales: Array,
    /// Per-group biases (`affine` only; `None` for `mxfp4`/`mxfp8`/`nvfp4`).
    quant_biases: Option<Array>,
    /// Optional `[output_dims]` output bias.
    bias: Option<Array>,
    /// Quantization group size.
    group_size: i32,
    /// Quantization bit depth.
    bits: i32,
    /// Quantization mode (`"affine"` / `"mxfp4"` / …).
    mode: String,
  },
}

impl BaseLinear {
  /// Build a dense base from a `[output_dims, input_dims]` weight (+ optional
  /// `[output_dims]` bias). Verifies rank-2 weight and matching bias shape.
  pub fn dense(weight: Array, bias: Option<Array>) -> Result<Self> {
    let w_shape = weight.shape();
    if w_shape.len() != 2 {
      return Err(Error::ShapeMismatch {
        message: format!(
          "BaseLinear::dense: weight must be 2-D [output_dims, input_dims], got {w_shape:?}"
        ),
      });
    }
    if let Some(b) = &bias {
      let b_shape = b.shape();
      if b_shape.len() != 1 || b_shape[0] != w_shape[0] {
        return Err(Error::ShapeMismatch {
          message: format!(
            "BaseLinear::dense: bias must be [output_dims={}], got {b_shape:?}",
            w_shape[0]
          ),
        });
      }
    }
    Ok(BaseLinear::Dense { weight, bias })
  }

  /// Build a quantized base from the MLX `(weight, scales, biases)` triple plus
  /// the scheme parameters. Validates the per-mode bias arity (mirroring
  /// [`crate::lm::nn::switch::QuantizedSwitchLinear::from_parts`]): `affine`
  /// REQUIRES `quant_biases`; the float schemes (`mxfp4`/`mxfp8`/`nvfp4`)
  /// forbid it.
  pub fn quantized(
    weight: Array,
    scales: Array,
    quant_biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: String,
  ) -> Result<Self> {
    match (mode.as_str(), &quant_biases) {
      ("affine", None) => {
        return Err(Error::Backend {
          message: "BaseLinear::quantized: `affine` mode requires quant_biases (mlx \
                    affine_quantize writes {w_q, scales, biases}), got None"
            .to_string(),
        });
      }
      ("mxfp4" | "mxfp8" | "nvfp4", Some(_)) => {
        return Err(Error::Backend {
          message: format!(
            "BaseLinear::quantized: `{mode}` mode is scale-only (mlx fp_quantize writes \
             {{w_q, scales}}), but quant_biases was provided"
          ),
        });
      }
      ("affine", Some(_)) | ("mxfp4" | "mxfp8" | "nvfp4", None) => {}
      (other, _) => {
        return Err(Error::Backend {
          message: format!(
            "BaseLinear::quantized: unknown mode {other:?}; allowed: \"affine\", \"mxfp4\", \
             \"mxfp8\", \"nvfp4\""
          ),
        });
      }
    }
    if bits <= 0 || group_size <= 0 {
      return Err(Error::Backend {
        message: format!(
          "BaseLinear::quantized: bits ({bits}) and group_size ({group_size}) must be > 0 \
           (per-mode value tables are validated by mlx-c)"
        ),
      });
    }
    Ok(BaseLinear::Quantized {
      weight,
      scales,
      quant_biases,
      bias,
      group_size,
      bits,
      mode,
    })
  }

  /// The optional output bias (the dense `[output_dims]` addend, NOT the
  /// quantization `biases`). `None` matches `bias=False`.
  pub fn bias(&self) -> Option<&Array> {
    match self {
      BaseLinear::Dense { bias, .. } => bias.as_ref(),
      BaseLinear::Quantized { bias, .. } => bias.as_ref(),
    }
  }

  /// The dense `[output_dims, input_dims]` weight, **dequantizing** if this is
  /// a quantized base (mlx-lm `_dequantized_weight`, `tuner/dora.py:92-106`).
  /// Used by `fuse` and the DoRA forward (which need the float weight to form
  /// the adapted magnitude).
  pub fn dequantized_weight(&self) -> Result<Array> {
    match self {
      BaseLinear::Dense { weight, .. } => weight.try_clone(),
      BaseLinear::Quantized {
        weight,
        scales,
        quant_biases,
        group_size,
        bits,
        mode,
        ..
      } => ops::quantized::dequantize(
        weight,
        scales,
        quant_biases.as_ref(),
        *group_size,
        *bits,
        mode,
        None,
        None,
      ),
    }
  }

  /// The base linear's output **without** the output bias: `x @ Wᵀ` for a dense
  /// base, a fused [`ops::quantized::quantized_matmul`] (`transpose=true`) for a
  /// quantized base. This is the bias-free base-output route the DoRA forward
  /// needs (mlx-lm `tuner/dora.py:113-114` / swift `QDoRALinear` `y = quantizedMM
  /// (...)` then `DoRALinear` `y = matmul(x, weight.T)`, `DoRA+Layers.swift:111,
  /// 172-174` — both bias-free, the bias is re-added after the magnitude renorm).
  ///
  /// Crucially, the quantized branch routes through `quantized_matmul` rather
  /// than dequantizing the full weight, so a QDoRA forward never materializes a
  /// dense `[output_dims, input_dims]` weight just to compute the base output.
  fn base_output_no_bias(&self, x: &Array) -> Result<Array> {
    match self {
      BaseLinear::Dense { weight, .. } => {
        let wt = weight.transpose()?;
        x.matmul(&wt)
      }
      BaseLinear::Quantized {
        weight,
        scales,
        quant_biases,
        group_size,
        bits,
        mode,
        ..
      } => {
        // `transpose=true` matches mlx-lm's QuantizedLinear (the packed weight
        // is laid out for the `output_dims x input_dims` orientation).
        ops::quantized::quantized_matmul(
          x,
          weight,
          scales,
          quant_biases.as_ref(),
          true,
          *group_size,
          *bits,
          mode,
        )
      }
    }
  }

  /// The base linear's output `y = x @ Wᵀ (+ bias)` — [`base_output_no_bias`]
  /// plus the optional output bias. Mirrors mlx-lm `self.linear(x)`
  /// (`tuner/lora.py:96`) / swift `super.callAsFunction(x)`. Does NOT add the
  /// low-rank term — that is [`LoRALinear::forward`]'s job.
  ///
  /// [`base_output_no_bias`]: BaseLinear::base_output_no_bias
  fn base_output(&self, x: &Array) -> Result<Array> {
    let y = self.base_output_no_bias(x)?;
    match self.bias() {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }

  /// Re-quantize a fused dense weight back into a [`BaseLinear::Quantized`]
  /// with this base's scheme — mlx-lm `nn.QuantizedLinear.from_linear(...)`
  /// in `fuse` (`tuner/lora.py:57-63`). Only meaningful for a quantized base;
  /// a dense base returns the dense fused linear unchanged.
  fn requantize_fused(&self, fused_weight: Array, fused_bias: Option<Array>) -> Result<BaseLinear> {
    match self {
      BaseLinear::Dense { .. } => BaseLinear::dense(fused_weight, fused_bias),
      BaseLinear::Quantized {
        group_size,
        bits,
        mode,
        ..
      } => {
        let (w_q, scales, q_biases) =
          ops::quantized::quantize(&fused_weight, *group_size, *bits, mode, None)?;
        BaseLinear::quantized(
          w_q,
          scales,
          q_biases,
          fused_bias,
          *group_size,
          *bits,
          mode.clone(),
        )
      }
    }
  }

  /// Whether this base is quantized (drives the `fuse(dequantize)` re-quantize
  /// decision).
  fn is_quantized(&self) -> bool {
    matches!(self, BaseLinear::Quantized { .. })
  }
}

// ─────────────────────── scalar-multiply helper ───────────────────────

/// `scale · arr`, broadcasting a scalar. MLX broadcasts a `[1]`-shaped array
/// against any shape, so a single `from_slice(&[scale], (1,))` × `multiply`
/// reproduces python's `scale * z` without an operator overload (mlxrs exposes
/// no `impl Mul`). Lazy — does not evaluate.
fn scaled(arr: &Array, scale: f32) -> Result<Array> {
  let s = Array::from_slice::<f32>(&[scale], &(1usize,))?;
  arr.multiply(&s)
}

/// `(scale · lora_bᵀ) @ lora_aᵀ` — the dense low-rank delta `Δ` of shape
/// `[output_dims, input_dims]`, the additive update shared by LoRA `fuse`
/// (mlx-lm `tuner/lora.py:52`) and the DoRA `adapted` weight
/// (`tuner/dora.py:120`). `lora_b` is `[r, output_dims]` → `lora_bᵀ` is
/// `[output_dims, r]`; `lora_a` is `[input_dims, r]` → `lora_aᵀ` is
/// `[r, input_dims]`; the product is `[output_dims, input_dims]`, matching the
/// base weight. Lazy.
fn lora_delta(params: &AdapterParams, scale: f32) -> Result<Array> {
  let lb_t = params.lora_b.transpose()?; // [output_dims, r]
  let la_t = params.lora_a.transpose()?; // [r, input_dims]
  let lb_t_scaled = scaled(&lb_t, scale)?;
  lb_t_scaled.matmul(&la_t)
}

/// The shared low-rank forward term `z = (x @ lora_a) @ lora_b`
/// (mlx-lm `tuner/lora.py:97` / `tuner/dora.py:116`), pre-scale. `x @ lora_a`
/// is `[..., r]`; `@ lora_b` is `[..., output_dims]`. Lazy.
fn lora_z(x: &Array, params: &AdapterParams) -> Result<Array> {
  let xa = x.matmul(&params.lora_a)?;
  xa.matmul(&params.lora_b)
}

// ──────────────────────────── LoRALinear ────────────────────────────

/// A LoRA-wrapped linear layer — mlx-lm `tuner/lora.py::LoRALinear` (dense
/// base) / its `QuantizedLinear` branch (quantized base, swift `QLoRALinear`).
///
/// Holds the [`BaseLinear`] (dense or quantized), the [`AdapterParams`]
/// (`lora_a` / `lora_b`), and the scalar `scale`. [`forward`](Self::forward)
/// adds the scaled low-rank update to the base output; [`fuse`](Self::fuse)
/// folds the update into the base weight.
///
/// Construct via [`LoRALinear::new`] (validates the factor shapes against the
/// base). The same type covers QLoRA (LoRA over a quantized base) — the
/// [`BaseLinear::Quantized`] variant routes the base output through a fused
/// quantized matmul (mlx-lm wraps `QuantizedLinear` with the *same* `LoRALinear`
/// class, `tuner/lora.py:22-23`).
#[derive(Debug)]
pub struct LoRALinear {
  base: BaseLinear,
  params: AdapterParams,
  scale: f32,
}

impl LoRALinear {
  /// Wrap `base` with the low-rank `params` and `scale`. Validates the factor
  /// shapes against the base dims: `lora_a` is `[input_dims, r]`, `lora_b` is
  /// `[r, output_dims]` (mlx-lm `tuner/lora.py:88-93`). A magnitude in
  /// `params` is ignored (use [`DoRALinear`] for the weight-decomposed forward).
  pub fn new(base: BaseLinear, params: AdapterParams, scale: f32) -> Result<Self> {
    validate_factor_shapes(&base, &params, "LoRALinear")?;
    Ok(Self {
      base,
      params,
      scale,
    })
  }

  /// The low-rank `scale` (mlx-lm `self.scale`).
  pub fn scale(&self) -> f32 {
    self.scale
  }

  /// The wrapped base linear.
  pub fn base(&self) -> &BaseLinear {
    &self.base
  }

  /// Forward pass `out = base(x) + scale · ((x @ lora_a) @ lora_b)` — mlx-lm
  /// `tuner/lora.py::LoRALinear.__call__` (`tuner/lora.py:95-98`) / swift
  /// `LoRALinear.callAsFunction`. The base output `base(x)` is `x @ Wᵀ (+
  /// bias)` (dense matmul or fused quantized matmul); the low-rank term adds
  /// `scale · z`. Lazy — does not evaluate.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let y = self.base.base_output(x)?;
    let z = lora_z(x, &self.params)?;
    let scaled_z = scaled(&z, self.scale)?;
    // mlx-lm casts the low-rank term back to x's dtype before the add
    // (`(self.scale * z).astype(x.dtype)`); replicate so a mixed-precision
    // base (e.g. fp16 weight, fp32 accumulation in the factors) matches.
    let scaled_z = match x.dtype() {
      Ok(dt) => scaled_z.astype(dt)?,
      Err(_) => scaled_z,
    };
    y.add(&scaled_z)
  }

  /// Fold the adapter into the base weight, returning a plain [`BaseLinear`]
  /// whose forward equals this layer's forward (the fusion is a no-op on the
  /// math). Mirrors mlx-lm `tuner/lora.py::LoRALinear.fuse` (`tuner/lora.py:34-65`)
  /// / swift `LoRALinear.fused()`.
  ///
  /// `W_fused = W + (scale · lora_bᵀ) @ lora_aᵀ`. For a quantized base the
  /// weight is dequantized, the delta added, then re-quantized with the same
  /// scheme unless `dequantize` is `true` (mlx-lm's `fuse(dequantize=...)`
  /// argument, `tuner/lora.py:34,57`), in which case the fused base is left
  /// dense.
  pub fn fuse(&self, dequantize: bool) -> Result<BaseLinear> {
    let weight = self.base.dequantized_weight()?;
    let delta = lora_delta(&self.params, self.scale)?;
    // mlx-lm casts the delta to the (dequantized) weight dtype before the add.
    let delta = match weight.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let fused_weight = weight.add(&delta)?;
    let fused_bias = match self.base.bias() {
      Some(b) => Some(b.try_clone()?),
      None => None,
    };
    if self.base.is_quantized() && !dequantize {
      self.base.requantize_fused(fused_weight, fused_bias)
    } else {
      BaseLinear::dense(fused_weight, fused_bias)
    }
  }
}

// ──────────────────────────── DoRALinear ────────────────────────────

/// A DoRA-wrapped linear layer — mlx-lm `tuner/dora.py::DoRALinear` (dense
/// base) / its `QuantizedLinear` branch (quantized base, swift `QDoRALinear`).
///
/// DoRA (Weight-Decomposed Low-Rank Adaptation) augments LoRA with a learned
/// per-output-row magnitude `m = ‖W‖₂ (axis 1)`, decoupling the weight's
/// direction (the LoRA-adapted, renormalized weight) from its magnitude.
/// Holds the [`BaseLinear`], the [`AdapterParams`] (`lora_a` / `lora_b` plus
/// the **required** `m`), and `scale`.
///
/// Construct via [`DoRALinear::new`] (validates the factor shapes AND requires
/// a magnitude). The same type covers QDoRA (DoRA over a quantized base) — the
/// base output runs through a fused quantized matmul (swift `QDoRALinear`,
/// `DoRA+Layers.swift:172-174`), and the dequantized weight is materialized
/// **only** for the adapted-weight L2-norm + fuse path (mlx-lm
/// `tuner/dora.py:92-106,120`).
#[derive(Debug)]
pub struct DoRALinear {
  base: BaseLinear,
  params: AdapterParams,
  magnitude: Array,
  scale: f32,
}

impl DoRALinear {
  /// Wrap `base` with the low-rank `params` and `scale` for the DoRA forward.
  /// Validates the factor shapes against the base dims and requires a
  /// magnitude `m` of shape `[output_dims]` in `params` (mlx-lm
  /// `tuner/dora.py:90`). The `m` is taken from `params.magnitude` (loaded
  /// from `adapters.safetensors`).
  pub fn new(base: BaseLinear, params: AdapterParams, scale: f32) -> Result<Self> {
    validate_factor_shapes(&base, &params, "DoRALinear")?;
    let magnitude = match &params.magnitude {
      Some(m) => m.try_clone()?,
      None => {
        return Err(Error::Backend {
          message: "DoRALinear::new: DoRA requires a magnitude `m` (loaded from \
                    adapters.safetensors), got None"
            .to_string(),
        });
      }
    };
    // `m` is the per-output-row norm: shape [output_dims].
    let output_dims = base_output_dims(&base)?;
    let m_shape = magnitude.shape();
    if m_shape.len() != 1 || m_shape[0] != output_dims {
      return Err(Error::ShapeMismatch {
        message: format!(
          "DoRALinear::new: magnitude `m` must be [output_dims={output_dims}], got {m_shape:?}"
        ),
      });
    }
    Ok(Self {
      base,
      params,
      magnitude,
      scale,
    })
  }

  /// The low-rank `scale`.
  pub fn scale(&self) -> f32 {
    self.scale
  }

  /// The wrapped base linear.
  pub fn base(&self) -> &BaseLinear {
    &self.base
  }

  /// The DoRA magnitude `m` (`[output_dims]`).
  pub fn magnitude(&self) -> &Array {
    &self.magnitude
  }

  /// Forward pass — mlx-lm `tuner/dora.py::DoRALinear.__call__`
  /// (`tuner/dora.py:111-128`) / swift `DoRALinear.callAsFunction` →
  /// `DoRA+Layers.swift::forward`:
  ///
  /// ```text
  /// y       = x @ Wᵀ            (base output, NO bias — quantized_matmul for a
  ///                              quantized base, never a dense dequantize)
  /// z       = (x @ lora_a) @ lora_b
  /// out     = y + (scale · z)
  /// w       = dequantized_weight (ONLY for the norm below)
  /// adapted = w + (scale · lora_bᵀ) @ lora_aᵀ
  /// denom   = ‖adapted‖₂ (axis 1)
  /// out     = (m / denom) · out  (+ bias)
  /// ```
  ///
  /// The renormalization `(m / denom)` is the weight-decomposition step that
  /// distinguishes DoRA from LoRA. For a quantized (QDoRA) base the base output
  /// `y` runs through [`ops::quantized::quantized_matmul`] (matching swift's
  /// `QDoRALinear` `y = quantizedMM(...)`, `DoRA+Layers.swift:172-174`) — the
  /// full weight is dequantized **only** to compute the adapted-weight L2-norm,
  /// never to form the base output, so a forward never materializes a dense
  /// `[output_dims, input_dims]` weight for the matmul. Lazy — does not evaluate.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    // y = base(x) WITHOUT the base bias (the bias is re-added at the very end,
    // mlx-lm `tuner/dora.py:113,126-127`, AFTER the magnitude renorm so it is
    // not scaled). Quantized base ⇒ quantized_matmul, NOT a dense dequantize.
    let y = self.base.base_output_no_bias(x)?;

    let z = lora_z(x, &self.params)?;
    let scaled_z = scaled(&z, self.scale)?;
    let scaled_z = match x.dtype() {
      Ok(dt) => scaled_z.astype(dt)?,
      Err(_) => scaled_z,
    };
    let out = y.add(&scaled_z)?;

    // adapted = w + (scale · lora_bᵀ) @ lora_aᵀ; denom = ‖adapted‖₂ (axis 1).
    // The dense weight is needed HERE (and only here) for the row-wise norm.
    let w = self.base.dequantized_weight()?;
    let delta = lora_delta(&self.params, self.scale)?;
    let delta = match w.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let adapted = w.add(&delta)?;
    // norm along axis 1 → [output_dims]; broadcasts against out's last axis.
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[1], false)?;
    let norm_scale = self.magnitude.divide(&denom)?;
    let norm_scale = match x.dtype() {
      Ok(dt) => norm_scale.astype(dt)?,
      Err(_) => norm_scale,
    };
    let mut out = out.multiply(&norm_scale)?;

    // Re-add the base bias AFTER the renorm (mlx-lm `tuner/dora.py:126-127`).
    if let Some(bias) = self.base.bias() {
      out = out.add(bias)?;
    }
    Ok(out)
  }

  /// Fold the DoRA adapter into the base weight — mlx-lm
  /// `tuner/dora.py::DoRALinear.fuse` (`tuner/dora.py:32-56`) / swift
  /// `DoRA+Layers.swift::fuse`:
  ///
  /// ```text
  /// W_adapted = w + (scale · lora_bᵀ) @ lora_aᵀ
  /// W_fused   = (m / ‖W_adapted‖₂)[:, None] · W_adapted
  /// ```
  ///
  /// The fused linear has **no** bias term folded into the weight (DoRA's
  /// `fuse` builds `nn.Linear(..., bias=False)` then re-attaches the original
  /// bias — `tuner/dora.py:38,46-47`). For a quantized base the weight is
  /// dequantized, fused, then re-quantized unless `dequantize` is `true`.
  pub fn fuse(&self, dequantize: bool) -> Result<BaseLinear> {
    let weight = self.base.dequantized_weight()?;
    let delta = lora_delta(&self.params, self.scale)?;
    let delta = match weight.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let adapted = weight.add(&delta)?;
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[1], false)?;
    let norm_scale = self.magnitude.divide(&denom)?;
    // norm_scale[:, None] — reshape [output_dims] → [output_dims, 1] so it
    // broadcasts down each weight row (mlx-lm `norm_scale[:, None] * weight`,
    // `tuner/dora.py:44`).
    let norm_scale_col = norm_scale.expand_dims_axes(&[-1])?;
    let fused_weight = norm_scale_col.multiply(&adapted)?;
    let fused_bias = match self.base.bias() {
      Some(b) => Some(b.try_clone()?),
      None => None,
    };
    if self.base.is_quantized() && !dequantize {
      self.base.requantize_fused(fused_weight, fused_bias)
    } else {
      BaseLinear::dense(fused_weight, fused_bias)
    }
  }
}

// ──────────────────────────── LoraLayer ────────────────────────────

/// A wrapped LoRA/DoRA linear layer — the unified runtime surface a
/// per-usecase architecture dispatches an adapted weight-path through. Mirrors
/// swift's `LoRALayer` protocol (`LoRA+Layers.swift` / `DoRA+Layers.swift`
/// both conform), which the [`LoraLayers`] map stores polymorphically.
///
/// Either [`LoraLayer::Lora`] or [`LoraLayer::Dora`], each carrying the
/// concrete [`LoRALinear`] / [`DoRALinear`]; [`forward`](Self::forward) and
/// [`fuse`](Self::fuse) forward to the variant.
#[derive(Debug)]
pub enum LoraLayer {
  /// A LoRA-wrapped linear.
  Lora(LoRALinear),
  /// A DoRA-wrapped linear.
  Dora(DoRALinear),
}

impl LoraLayer {
  /// Forward through the wrapped layer (LoRA or DoRA). Lazy.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    match self {
      LoraLayer::Lora(l) => l.forward(x),
      LoraLayer::Dora(d) => d.forward(x),
    }
  }

  /// Fuse the wrapped layer into a plain [`BaseLinear`] (see
  /// [`LoRALinear::fuse`] / [`DoRALinear::fuse`]).
  pub fn fuse(&self, dequantize: bool) -> Result<BaseLinear> {
    match self {
      LoraLayer::Lora(l) => l.fuse(dequantize),
      LoraLayer::Dora(d) => d.fuse(dequantize),
    }
  }

  /// The wrapped base linear.
  pub fn base(&self) -> &BaseLinear {
    match self {
      LoraLayer::Lora(l) => l.base(),
      LoraLayer::Dora(d) => d.base(),
    }
  }
}

/// The map a [`linear_to_lora_layers`] / [`load_adapters`] run produces: the
/// base-weight **path** (e.g. `"model.layers.27.self_attn.q_proj"`) → its
/// wrapped [`LoraLayer`].
///
/// This is the weight-map analogue of the in-place `nn.Module` replacement
/// mlx-lm / swift perform — a per-usecase architecture that already routes a
/// path to its forward call dispatches through the wrapped layer for any path
/// present in this map (and leaves un-adapted paths on their base forward).
pub type LoraLayers = HashMap<String, LoraLayer>;

// ──────────────────── shape validation helpers ────────────────────

/// The base linear's `output_dims` (the leading weight dim for a dense base;
/// for a quantized base the *packed* weight's leading dim still equals
/// `output_dims` — MLX packs along the last axis only).
fn base_output_dims(base: &BaseLinear) -> Result<usize> {
  let shape = match base {
    BaseLinear::Dense { weight, .. } => weight.shape(),
    BaseLinear::Quantized { weight, .. } => weight.shape(),
  };
  shape.first().copied().ok_or_else(|| Error::ShapeMismatch {
    message: "base linear weight has rank 0; cannot determine output_dims".to_string(),
  })
}

/// The base linear's `input_dims` — the contraction dimension `lora_a`'s leading
/// axis must equal. For a **dense** base it is the weight's trailing axis
/// (`weight` is `[output_dims, input_dims]`). For a **quantized** base the
/// *packed* weight's trailing axis is `input_dims * bits / 32` (MLX packs
/// `32 / bits` weights per `uint32` along the last axis), so the logical input
/// width is `packed_last_axis * 32 / bits` — exactly mlx-lm's `from_base`
/// recovery `input_dims = input_dims * 32 // bits` (`tuner/lora.py:23`,
/// `tuner/dora.py:21`). `bits` is validated `> 0` by [`BaseLinear::quantized`].
fn base_input_dims(base: &BaseLinear) -> Result<usize> {
  match base {
    BaseLinear::Dense { weight, .. } => {
      let shape = weight.shape();
      shape.get(1).copied().ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "dense base weight must be 2-D [output_dims, input_dims]; got rank-{} shape {shape:?}",
          shape.len()
        ),
      })
    }
    BaseLinear::Quantized { weight, bits, .. } => {
      let shape = weight.shape();
      let packed = shape.get(1).copied().ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "quantized base weight must be 2-D [output_dims, input_dims*bits/32]; got rank-{} \
           shape {shape:?}",
          shape.len()
        ),
      })?;
      // `bits > 0` is guaranteed by `BaseLinear::quantized`; recover the logical
      // input width `packed * 32 / bits` (e.g. 4-bit packs 8 weights / u32).
      Ok(packed * 32 / (*bits as usize))
    }
  }
}

/// Validate `lora_a` / `lora_b` against the base dims. `lora_a` is
/// `[input_dims, r]`, so its leading axis must equal the base `input_dims`
/// (recovered from the packed width for a quantized base — see
/// [`base_input_dims`]) and its last axis (`r`) must match `lora_b`'s leading
/// axis (`r`); `lora_b` is `[r, output_dims]`, so its last axis must equal the
/// base `output_dims`. Cross-checking the `input_dims` axis here means a wrong
/// `lora_a` width is a recoverable [`Error::ShapeMismatch`] at validate/load
/// time (not an opaque mlx-c matmul failure on the first forward).
fn validate_factor_shapes(base: &BaseLinear, params: &AdapterParams, who: &str) -> Result<()> {
  let a_shape = params.lora_a.shape();
  let b_shape = params.lora_b.shape();
  if a_shape.len() != 2 {
    return Err(Error::ShapeMismatch {
      message: format!("{who}: lora_a must be 2-D [input_dims, r], got {a_shape:?}"),
    });
  }
  if b_shape.len() != 2 {
    return Err(Error::ShapeMismatch {
      message: format!("{who}: lora_b must be 2-D [r, output_dims], got {b_shape:?}"),
    });
  }
  // r consistency: lora_a's last axis == lora_b's leading axis.
  if a_shape[1] != b_shape[0] {
    return Err(Error::ShapeMismatch {
      message: format!(
        "{who}: rank mismatch — lora_a is [_, r={}] but lora_b is [r={}, _]",
        a_shape[1], b_shape[0]
      ),
    });
  }
  // input_dims consistency: lora_a's leading axis == base input_dims.
  let input_dims = base_input_dims(base)?;
  if a_shape[0] != input_dims {
    return Err(Error::ShapeMismatch {
      message: format!(
        "{who}: lora_a leading axis ({}) must equal base input_dims ({input_dims})",
        a_shape[0]
      ),
    });
  }
  // output_dims consistency: lora_b's last axis == base output_dims.
  let output_dims = base_output_dims(base)?;
  if b_shape[1] != output_dims {
    return Err(Error::ShapeMismatch {
      message: format!(
        "{who}: lora_b last axis ({}) must equal base output_dims ({output_dims})",
        b_shape[1]
      ),
    });
  }
  Ok(())
}

/// Check the adapter factor tensors' rank axis against the rank declared in
/// `adapter_config.json` (`config.rank()`) — the *config-vs-tensor* rank
/// cross-check.
///
/// [`validate_factor_shapes`] only verifies `lora_a` and `lora_b` agree with
/// **each other** on the shared rank axis; it cannot see the config. But the
/// layer SCALE is `alpha / config.rank()` when an `alpha` (`lora_alpha`) is
/// present, so a config whose `rank` has drifted from the tensors' rank — a
/// stale `adapter_config.json`, or a PEFT config whose `r` key was not
/// recognized and silently defaulted — would otherwise build rank-`R` tensors
/// while scaling by `alpha / config.rank()` (the wrong divisor): silently wrong
/// strength on every adapted projection.
///
/// Requiring `lora_a`'s rank axis (`[input_dims, r]`, last axis) and
/// `lora_b`'s rank axis (`[r, output_dims]`, leading axis) to both equal
/// `config_rank` makes that drift a loud, recoverable [`Error::ShapeMismatch`]
/// at load time instead. Indexing is defensive (a non-2-D factor reads as a
/// `0` rank axis), so this is safe to call independently of
/// [`validate_factor_shapes`].
fn validate_config_rank(params: &AdapterParams, config_rank: usize, who: &str) -> Result<()> {
  let a_shape = params.lora_a.shape();
  let b_shape = params.lora_b.shape();
  // A well-formed `lora_a` is `[input_dims, r]` and `lora_b` is
  // `[r, output_dims]`; a non-2-D factor reads as a `0` rank axis here and
  // fails the equality below (it also fails `validate_factor_shapes`).
  let a_rank = a_shape.get(1).copied().unwrap_or_default();
  let b_rank = b_shape.first().copied().unwrap_or_default();
  if a_rank != config_rank || b_rank != config_rank {
    return Err(Error::ShapeMismatch {
      message: format!(
        "{who}: adapter factor rank ({a_rank}) does not match adapter_config.json rank \
         ({config_rank}); a stale config (or a PEFT config whose `r` key was not recognized) \
         would silently scale by alpha/{config_rank} instead of alpha/{a_rank}"
      ),
    });
  }
  Ok(())
}

// ─────────────────────── linear_to_lora_layers ───────────────────────

/// Apply LoRA/DoRA wrapping to the targeted linear layers of a [`Weights`]
/// map — mlx-lm `tuner/utils.py::linear_to_lora_layers` (`tuner/utils.py:38-110`)
/// adapted to the weight-map model (see the [module docs](self)).
///
/// For each base-weight path the predicate selects, this builds a [`LoraLayer`]
/// (LoRA or DoRA per `config`) over the path's [`BaseLinear`] — a dense base
/// from `<path>.weight` (+ optional `<path>.bias`), or a quantized base from
/// the `<path>.weight` / `<path>.scales` / `<path>.biases` triple when `quant`
/// resolves a [`Quantization`] for that path (the QLoRA case). The returned
/// [`LoraLayers`] map carries the wrapped layers; un-targeted paths are not
/// touched.
///
/// # Layer selection
///
/// Faithful to mlx-lm's two-part predicate:
///
/// - **`num_layers`** — only the **last** `num_layers` decoder blocks are
///   adapted (mlx-lm `model.layers[-max(num_layers, 0):]`,
///   `tuner/utils.py:103`), EXCEPT that a **non-positive** `num_layers` selects
///   ALL blocks (the Python `-0` slice quirk: `layers[-0:]` == `layers[0:]`),
///   so `num_layers: -1` (and `0`) adapt every block. A path's block index is
///   parsed from the
///   `…layers.N.…` segment; a path with no such segment (e.g. a top-level
///   `lm_head`) is adapted only when it matches `keys` AND `num_layers`
///   covers all blocks is not applicable to it — to stay faithful, non-block
///   paths are wrapped only when `keys` is explicit and the path matches (the
///   per-layer-block window does not gate a non-block path).
/// - **`keys`** — when `config.lora_parameters.keys` is set, a path is adapted
///   only if it **ends with** one of the keys (e.g. key `"self_attn.q_proj"`
///   matches `"model.layers.27.self_attn.q_proj"`), mirroring mlx-lm's
///   `k in keys` module-name match (`tuner/utils.py:104`). When `keys` is
///   `None`, every eligible linear in the window is adapted (mlx-lm's
///   auto-discovery, `tuner/utils.py:85-101`); a weight is "an eligible
///   linear" when its `<path>.weight` is rank-2 (the structural analogue of
///   mlx-lm's `isinstance(layer, nn.Linear)` — the same rank-2 gate
///   [`crate::lm::quant`] uses).
///
/// `adapter_params` supplies the per-path [`AdapterParams`] (loaded from
/// `adapters.safetensors`). The total number of decoder blocks (`num_blocks`)
/// is needed to resolve the trailing-`num_layers` window; pass the model's
/// layer count (mlx-lm reads `len(model.layers)`).
///
/// # Completeness postcondition
///
/// After wrapping, the result is checked (`check_adapter_completeness`) so a
/// path-prefix mismatch / missing tensor group / empty safetensors /
/// `adapter_config.json` drift cannot silently return a partially- or
/// un-adapted model. It is a recoverable [`Error::Backend`] when (a) an
/// explicit `keys` selection is missing factors for a selected target, (b) an
/// `adapter_params` factor group matches no base layer, or (c) nothing was
/// adapted at all.
///
/// A selected path whose factor shapes don't match the base (or a DoRA path
/// with no magnitude) is a recoverable [`Error::ShapeMismatch`] /
/// [`Error::Backend`]. A selected path whose factor tensors' rank axis
/// disagrees with `config.rank()` (a stale `adapter_config.json`, or a PEFT
/// `r` key that defaulted) is a recoverable [`Error::ShapeMismatch`]
/// (`validate_config_rank`) — caught before the `alpha / rank` scale is
/// applied, so a rank drift cannot silently scale by the wrong divisor.
pub fn linear_to_lora_layers(
  weights: &Weights,
  config: &LoraConfig,
  adapter_params: &HashMap<String, AdapterParams>,
  quant: Option<&PerLayerQuantization>,
  num_blocks: i32,
) -> Result<LoraLayers> {
  let mut out: LoraLayers = HashMap::new();
  let scale = config.scale();
  let is_dora = config.is_dora();
  let keys = config.lora_parameters.keys.as_deref();
  // The config-declared rank, cross-checked against every adapter factor group
  // below (`validate_config_rank`). A non-positive rank cannot index a tensor
  // axis — `load_adapters` already rejects it, but `linear_to_lora_layers` is
  // also a public entry point, so guard here too. `None` ⇒ skip the
  // config-rank cross-check (degenerate config; the empty/zero-rank factors it
  // would build are caught by the shape checks instead).
  let config_rank: Option<usize> = usize::try_from(config.rank()).ok().filter(|&r| r > 0);
  // mlx-lm selects `model.layers[-max(num_layers, 0):]` (`tuner/utils.py:103`).
  // Note the Python `-0` quirk: when `num_layers <= 0`, `max(num_layers, 0)` is
  // `0` and `layers[-0:]` == `layers[0:]` == ALL blocks (so `num_layers: -1` —
  // and `0` — selects every block, NOT none). For `num_layers > 0` it is the
  // trailing `num_layers` blocks. Reproduce: a non-positive `num_layers` starts
  // the window at block 0 (all blocks); a positive one at `num_blocks -
  // num_layers`.
  let first_adapted = if config.num_layers <= 0 {
    0
  } else {
    (num_blocks - config.num_layers).max(0)
  };

  // Completeness tracking (the postcondition below): every adapter factor
  // group MUST be applied to a base layer, and an explicit `keys` selection
  // MUST find its factors — otherwise a path-prefix mismatch / missing tensor
  // group / config drift would silently yield a partially- or un-adapted model.
  let mut consumed: HashSet<&str> = HashSet::with_capacity(adapter_params.len());
  // Targets the predicate selected but for which no factors were supplied —
  // only an *error* when `keys` is explicit (with auto-discovery, an unmatched
  // linear is expected — the adapter legitimately trains only a subset).
  let mut selected_without_factors: Vec<&str> = Vec::new();

  for (key, weight) in weights {
    let Some(path) = key.strip_suffix(".weight") else {
      continue;
    };

    // keys filter (suffix match) — or rank-2 auto-discovery when keys is None.
    if let Some(keys) = keys {
      if !keys.iter().any(|k| path_matches_key(path, k)) {
        continue;
      }
    } else if weight.shape().len() != 2 {
      // auto-discovery: only rank-2 weights are "linears".
      continue;
    }

    // num_layers window: a path inside a decoder block is adapted only when
    // its block index is in the trailing window. A non-block path (no
    // `layers.N`) is governed by `keys` alone (it has no block index to gate).
    if let Some(block) = parse_block_index(path)
      && block < first_adapted
    {
      continue;
    }

    // `path` is now a SELECTED target (predicate-matched). Build a layer only
    // when we actually have factors for it; record a missing-factor target so
    // the postcondition can reject an incomplete explicit-`keys` selection.
    let Some(params) = adapter_params.get(path) else {
      selected_without_factors.push(path);
      continue;
    };
    consumed.insert(path);

    // Cross-check the factor tensors' rank axis against the config-declared
    // rank BEFORE building the layer (and resolving the alpha/rank scale): a
    // config/tensor rank drift must fail loudly here, not silently scale by
    // the wrong divisor (`alpha / config.rank()`).
    if let Some(rank) = config_rank {
      let who = if is_dora { "DoRALinear" } else { "LoRALinear" };
      validate_config_rank(params, rank, who)?;
    }

    let base = build_base_linear(weights, path, weight, quant)?;
    let layer = if is_dora {
      LoraLayer::Dora(DoRALinear::new(base, params.try_clone()?, scale)?)
    } else {
      LoraLayer::Lora(LoRALinear::new(base, params.try_clone()?, scale)?)
    };
    out.insert(path.to_string(), layer);
  }

  check_adapter_completeness(
    &out,
    adapter_params,
    &consumed,
    &selected_without_factors,
    keys,
  )?;
  Ok(out)
}

/// The adapter-completeness postcondition for [`linear_to_lora_layers`]:
/// reject a result that would leave inference silently-wrong.
///
/// A base path matching the `keys`/`num_layers` predicate but carrying no
/// [`AdapterParams`] used to be skipped silently — a path-prefix mismatch,
/// missing tensor group, empty `adapters.safetensors`, or `adapter_config.json`
/// drift would then return `Ok` with a partially- or un-adapted model. This
/// catches all three failure modes:
///
/// - **(a) explicitly-selected target with no factors** — when `keys` is an
///   explicit list, every `(key × in-window-block)` path is a target the
///   adapter is expected to provide; a missing factor group is config drift.
///   (With `keys: None` auto-discovery an unmatched linear is *expected* — the
///   adapter trains only a subset — so this is not checked there.)
/// - **(b) unused adapter factor group** — every path present in
///   `adapter_params` (i.e. every `<path>.lora_a`/`lora_b` group in the
///   safetensors) MUST have matched a base layer; one that matched nothing is a
///   path-prefix mismatch. This is the analogue of swift's
///   `model.update(parameters:, verify: .noUnusedKeys)` (`LoRAContainer.swift:152`).
/// - **(c) empty result** — no layer adapted at all ⇒ the adapter did nothing.
///
/// Each violation is a recoverable [`Error::Backend`] naming the offending
/// paths.
fn check_adapter_completeness(
  applied: &LoraLayers,
  adapter_params: &HashMap<String, AdapterParams>,
  consumed: &HashSet<&str>,
  selected_without_factors: &[&str],
  keys: Option<&[String]>,
) -> Result<()> {
  // (a) explicit `keys` selection that is missing factors.
  if keys.is_some() && !selected_without_factors.is_empty() {
    let mut missing: Vec<&str> = selected_without_factors.to_vec();
    missing.sort_unstable();
    return Err(Error::Backend {
      message: format!(
        "load_adapters: adapter is missing factors for {} explicitly-selected target(s): {:?}; \
         the adapter_config.json `keys`/`num_layers` selection does not match the \
         adapters.safetensors contents",
        missing.len(),
        missing
      ),
    });
  }

  // (b) adapter factor groups that matched no base layer (unused).
  let mut unused: Vec<&str> = adapter_params
    .keys()
    .map(String::as_str)
    .filter(|p| !consumed.contains(p))
    .collect();
  if !unused.is_empty() {
    unused.sort_unstable();
    return Err(Error::Backend {
      message: format!(
        "load_adapters: {} adapter factor group(s) match no base layer: {:?}; the \
         adapters.safetensors paths do not line up with the base model weights (path-prefix \
         mismatch or config drift)",
        unused.len(),
        unused
      ),
    });
  }

  // (c) nothing adapted at all.
  if applied.is_empty() {
    return Err(Error::Backend {
      message: "load_adapters: no base layer was adapted — the adapter_config.json \
                `keys`/`num_layers` selection matched nothing in the base model, or \
                adapters.safetensors carried no factors"
        .to_string(),
    });
  }

  Ok(())
}

/// Build the [`BaseLinear`] for `path` from the weight map: a quantized base
/// (from the `<path>.weight` / `.scales` / `.biases` triple) when `quant`
/// resolves a [`Quantization`] for `path` AND a `<path>.scales` sibling
/// exists; otherwise a dense base (from `<path>.weight` + optional
/// `<path>.bias`).
fn build_base_linear(
  weights: &Weights,
  path: &str,
  weight: &Array,
  quant: Option<&PerLayerQuantization>,
) -> Result<BaseLinear> {
  let scales_key = format!("{path}.scales");
  let biases_key = format!("{path}.biases");
  let bias_key = format!("{path}.bias");

  // The QLoRA case: a resolvable Quantization for this path AND a `.scales`
  // sibling (the load-bearing quantized-layout signal — mlx-lm's
  // `f"{p}.scales" in weights` check, `utils.py:349-355`).
  let q: Option<Quantization> = quant.and_then(|c| c.quantization_for(path));
  if let (Some(q), Some(scales)) = (q, weights.get(&scales_key)) {
    let quant_biases = weights.get(&biases_key).map(Array::try_clone).transpose()?;
    let bias = weights.get(&bias_key).map(Array::try_clone).transpose()?;
    return BaseLinear::quantized(
      weight.try_clone()?,
      scales.try_clone()?,
      quant_biases,
      bias,
      q.group_size,
      q.bits,
      q.mode.as_mlx_str().to_string(),
    );
  }

  // Dense base.
  let bias = weights.get(&bias_key).map(Array::try_clone).transpose()?;
  BaseLinear::dense(weight.try_clone()?, bias)
}

/// Whether `path` should match the adapter key `key`: mlx-lm matches a module
/// **name** (the path tail). A `key` matches when `path` equals it or ends
/// with `".{key}"` (so `"self_attn.q_proj"` matches
/// `"model.layers.27.self_attn.q_proj"` but not `"…xself_attn.q_proj"`).
fn path_matches_key(path: &str, key: &str) -> bool {
  path == key || path.ends_with(&format!(".{key}"))
}

/// Parse the decoder-block index from a `…layers.N.…` (or trailing
/// `…layers.N`) path segment, mirroring mlx-lm's per-block iteration over
/// `model.layers`. `None` when there is no `layers.<int>` segment (a
/// non-block path — e.g. `model.embed_tokens` / `lm_head`).
fn parse_block_index(path: &str) -> Option<i32> {
  // Find a "layers." segment and parse the following integer up to the next
  // '.' or end of string.
  let marker = "layers.";
  let idx = path.find(marker)? + marker.len();
  let rest = &path[idx..];
  let end = rest.find('.').unwrap_or(rest.len());
  rest[..end].parse::<i32>().ok()
}

// ───────────────────────────── load_adapters ─────────────────────────────

/// Load a pre-trained adapter from a **local** directory and apply it to a
/// base model's [`Weights`] map — mlx-lm `tuner/utils.py::load_adapters`
/// (`tuner/utils.py:113-138`) + swift `LoRAContainer.from(directory:)` /
/// `load(into:)`, restricted to the local-path, no-network surface.
///
/// Reads `<dir>/adapter_config.json` (bounded, untrusted-dir-safe — same
/// discipline as [`crate::lm::load::load_config`]) and
/// `<dir>/adapters.safetensors` (via [`crate::io::load_safetensors`]), splits
/// the safetensors into per-path [`AdapterParams`] (`<path>.lora_a` /
/// `<path>.lora_b` / `<path>.m`), then runs [`linear_to_lora_layers`] over
/// `base_weights` to build the [`LoraLayers`] map.
///
/// `base_weights` is the loaded base-model weight map ([`crate::lm::load::load_weights`]);
/// `quant` is the base model's [`PerLayerQuantization`] (from
/// [`crate::lm::quant::parse_quantization`] on the base `config.json`) so a
/// quantized base routes through the QLoRA path — pass `None` for a dense base.
/// `num_blocks` is the base model's decoder-block count.
///
/// # Errors (recoverable)
///
/// - Missing adapter dir / `adapter_config.json` / `adapters.safetensors`,
///   oversized / non-regular / non-UTF-8 config → [`Error::Backend`].
/// - An `adapters.safetensors` that is not a regular file (FIFO / device /
///   directory) or exceeds [`MAX_ADAPTER_SAFETENSORS_BYTES`] → [`Error::Backend`]
///   (the file is stat-checked before mlx-c mmaps it).
/// - `fine_tune_type: "full"` (a full-weight fine-tune, not an adapter — see
///   [`FineTuneType::Full`]) → [`Error::Backend`] (unsupported here).
///   An **unknown** `fine_tune_type` string is a serde parse error →
///   [`Error::Backend`] from [`LoraConfig::from_json`].
/// - A target path with a magnitude-less DoRA factor, or factor shapes that
///   don't match the base → [`Error::ShapeMismatch`] / [`Error::Backend`].
/// - The completeness postcondition of [`linear_to_lora_layers`]: an explicit
///   `keys` selection missing factors, an unused adapter factor group, or an
///   empty result → [`Error::Backend`].
pub fn load_adapters(
  base_weights: &Weights,
  dir: &Path,
  quant: Option<&PerLayerQuantization>,
  num_blocks: i32,
) -> Result<LoraLayers> {
  // 1) adapter_config.json (bounded read, untrusted-dir-safe).
  let config_text = read_bounded_adapter_config(dir)?;
  let config = LoraConfig::from_json(&config_text)?;

  // mlx-lm skips linear_to_lora_layers for "full" and loads a dense delta;
  // mlxrs has no module tree to merge a full fine-tune into here, so reject it
  // as unsupported (recoverable) — the per-usecase architecture merges a full
  // fine-tune at the weight-map level instead.
  if config.fine_tune_type == FineTuneType::Full {
    return Err(Error::Backend {
      message: "load_adapters: fine_tune_type \"full\" is a full-weight fine-tune, not an \
                adapter; merge it at the weight-map level via lm::load::load_weights instead"
        .to_string(),
    });
  }

  // Reject a non-positive rank early (a degenerate config that would build
  // empty factors).
  if config.rank() <= 0 {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: adapter rank must be > 0, got {}",
        config.rank()
      ),
    });
  }

  // 2) adapters.safetensors → per-path AdapterParams. Stat the file FIRST
  // (regular-file + size-budget) so an untrusted adapter dir cannot point us at
  // a FIFO/device (hang/opaque error) or an oversized blob (OOM) — the
  // safetensors path is otherwise handed straight to mlx-c, which would mmap
  // whatever it is given.
  let st_path = dir.join("adapters.safetensors");
  check_adapter_safetensors(&st_path)?;
  let adapter_arrays = crate::io::load_safetensors(&st_path).map_err(|e| Error::Backend {
    message: format!("load_adapters: cannot load {}: {e}", st_path.display()),
  })?;
  let adapter_params = split_adapter_params(adapter_arrays, config.is_dora())?;

  // 3) Build + apply the LoRA/DoRA layers over the base weight map.
  linear_to_lora_layers(base_weights, &config, &adapter_params, quant, num_blocks)
}

/// Split the flat `adapters.safetensors` array map into per-path
/// [`AdapterParams`]. mlx-lm's `LoRALinear` registers its factors at
/// `<module>.lora_a` / `<module>.lora_b` (and DoRA's `<module>.m`), so the
/// safetensors keys are `<path>.lora_a` / `<path>.lora_b` / `<path>.m`. Groups
/// by stripping those suffixes; a `.lora_a` without a matching `.lora_b` (or
/// vice versa) is a recoverable [`Error::Backend`]. When `expect_dora`, a
/// group missing its `.m` is an error; when not, a stray `.m` is ignored.
fn split_adapter_params(
  arrays: HashMap<String, Array>,
  expect_dora: bool,
) -> Result<HashMap<String, AdapterParams>> {
  // Collect the three slots per path.
  let mut a_map: HashMap<String, Array> = HashMap::new();
  let mut b_map: HashMap<String, Array> = HashMap::new();
  let mut m_map: HashMap<String, Array> = HashMap::new();

  for (key, arr) in arrays {
    if let Some(path) = key.strip_suffix(".lora_a") {
      a_map.insert(path.to_string(), arr);
    } else if let Some(path) = key.strip_suffix(".lora_b") {
      b_map.insert(path.to_string(), arr);
    } else if let Some(path) = key.strip_suffix(".m") {
      m_map.insert(path.to_string(), arr);
    }
    // Any other key (e.g. a saved base weight in a "full" checkpoint) is
    // ignored — this path only handles low-rank adapters.
  }

  let mut out: HashMap<String, AdapterParams> = HashMap::with_capacity(a_map.len());
  for (path, lora_a) in a_map {
    let lora_b = b_map.remove(&path).ok_or_else(|| Error::Backend {
      message: format!(
        "load_adapters: adapter path {path:?} has `lora_a` but no matching `lora_b`"
      ),
    })?;
    let magnitude = m_map.remove(&path);
    if expect_dora && magnitude.is_none() {
      return Err(Error::Backend {
        message: format!("load_adapters: DoRA adapter path {path:?} is missing its magnitude `m`"),
      });
    }
    out.insert(
      path,
      AdapterParams {
        lora_a,
        lora_b,
        magnitude,
      },
    );
  }

  // Any `lora_b` left without a matching `lora_a` is an error.
  if let Some((path, _)) = b_map.into_iter().next() {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: adapter path {path:?} has `lora_b` but no matching `lora_a`"
      ),
    });
  }

  Ok(out)
}

/// Read `<dir>/adapter_config.json` with the same bounded, untrusted-dir-safe
/// discipline as [`crate::lm::load::load_config`]: open once (closing the
/// stat-then-read TOCTOU window), reject a non-regular file before any read,
/// cap the body at [`crate::lm::load::MAX_CONFIG_BYTES`] via `Read::take`, and
/// on Unix carry `O_NONBLOCK | O_CLOEXEC` so a planted FIFO returns
/// immediately. Every failure (missing dir/file, non-regular, oversized,
/// unreadable, non-UTF-8) is a recoverable [`Error::Backend`].
fn read_bounded_adapter_config(dir: &Path) -> Result<String> {
  use std::io::Read;

  if !dir.exists() {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: adapter path does not exist: {}",
        dir.display()
      ),
    });
  }
  let path = dir.join("adapter_config.json");

  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(&path)
      .map_err(|e| Error::Backend {
        message: format!("load_adapters: cannot open {}: {e}", path.display()),
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(&path).map_err(|e| Error::Backend {
    message: format!("load_adapters: cannot open {}: {e}", path.display()),
  })?;

  let meta = file.metadata().map_err(|e| Error::Backend {
    message: format!("load_adapters: cannot stat {}: {e}", path.display()),
  })?;
  if !meta.is_file() {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: {} is not a regular file; refusing to read",
        path.display()
      ),
    });
  }

  let cap = crate::lm::load::MAX_CONFIG_BYTES;
  let mut bytes = Vec::new();
  file
    .take(cap + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| Error::Backend {
      message: format!("load_adapters: cannot read {}: {e}", path.display()),
    })?;
  if bytes.len() as u64 > cap {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: {} exceeds the {cap}-byte cap; refusing to read",
        path.display()
      ),
    });
  }
  String::from_utf8(bytes).map_err(|e| Error::Backend {
    message: format!("load_adapters: {} is not valid UTF-8: {e}", path.display()),
  })
}

/// Stat `<dir>/adapters.safetensors` before it is handed to
/// [`crate::io::load_safetensors`] (which mmaps whatever path it is given,
/// performing no validation). Mirrors the regular-file discipline of
/// [`read_bounded_adapter_config`] / [`crate::lm::load`]'s shard discovery:
///
/// - Open once with `O_NONBLOCK | O_CLOEXEC` on Unix so a planted **FIFO**
///   returns immediately instead of blocking the caller (symlinks are followed
///   — a cached-model layout may symlink the file — but the post-open `fstat`
///   below checks the *resolved target*).
/// - `fstat` the opened handle and require a **regular file**: a FIFO / device
///   / directory / symlink-to-any-of-those is rejected before `load_safetensors`
///   can mmap it.
/// - Enforce the [`MAX_ADAPTER_SAFETENSORS_BYTES`] budget on the reported size
///   so an oversized blob is a clear recoverable error, not an OOM.
///
/// Every violation (missing file, non-regular, oversized, unstattable) is a
/// recoverable [`Error::Backend`]. The handle is closed on return; the
/// subsequent [`crate::io::load_safetensors`] re-opens via mlx-c. (This leaves
/// a narrow TOCTOU window between the check and mlx-c's open — acceptable here,
/// matching `lm::load`'s shard discovery, since `load_safetensors` cannot be
/// handed a pre-opened descriptor; the budget still bounds a same-size swap and
/// `O_NONBLOCK` is moot once a regular file has been confirmed.)
fn check_adapter_safetensors(path: &Path) -> Result<()> {
  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(path)
      .map_err(|e| Error::Backend {
        message: format!("load_adapters: cannot open {}: {e}", path.display()),
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(path).map_err(|e| Error::Backend {
    message: format!("load_adapters: cannot open {}: {e}", path.display()),
  })?;

  let meta = file.metadata().map_err(|e| Error::Backend {
    message: format!("load_adapters: cannot stat {}: {e}", path.display()),
  })?;
  if !meta.is_file() {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: {} is not a regular file; refusing to load",
        path.display()
      ),
    });
  }
  if meta.len() > MAX_ADAPTER_SAFETENSORS_BYTES {
    return Err(Error::Backend {
      message: format!(
        "load_adapters: {} is {} bytes, exceeding the {MAX_ADAPTER_SAFETENSORS_BYTES}-byte \
         adapter budget; refusing to load",
        path.display(),
        meta.len()
      ),
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  // ───────────────────── hand-traced fixtures ─────────────────────

  /// Base weight `W` of shape [output_dims=2, input_dims=3]:
  /// ```text
  /// [[1, 0, 0],
  ///  [0, 1, 0]]
  /// ```
  /// so `x @ Wᵀ` projects `x=[x0,x1,x2]` to `[x0, x1]`.
  fn base_weight() -> Array {
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap()
  }

  /// `lora_a` of shape [input_dims=3, r=2]:
  /// ```text
  /// [[1, 0],
  ///  [0, 1],
  ///  [0, 0]]
  /// ```
  fn lora_a() -> Array {
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap()
  }

  /// `lora_b` of shape [r=2, output_dims=2]:
  /// ```text
  /// [[1, 0],
  ///  [0, 1]]
  /// ```
  fn lora_b() -> Array {
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap()
  }

  fn plain_params() -> AdapterParams {
    AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: None,
    }
  }

  fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b.iter()) {
      assert!((x - y).abs() <= tol, "‖{x} - {y}‖ > {tol} ({a:?} vs {b:?})");
    }
  }

  // ───────────────────── LoRALinear forward ─────────────────────

  #[test]
  fn lora_linear_forward_hand_traced() {
    // x = [1, 2, 3]; scale = 2.0.
    // base(x)  = x @ Wᵀ = [1, 2]
    // x @ a    = [1, 2]  (a picks first two coords)
    // (x@a)@b  = [1, 2]
    // out      = base + scale*z = [1 + 2*1, 2 + 2*2] = [3, 6]
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
  }

  #[test]
  fn lora_linear_forward_with_bias() {
    // bias = [10, 20]; out = [3, 6] + [10, 20] = [13, 26].
    let bias = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap();
    let base = BaseLinear::dense(base_weight(), Some(bias)).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[13.0, 26.0], 1e-5);
  }

  #[test]
  fn lora_linear_zero_b_is_identity() {
    // lora_b all zeros ⇒ the low-rank term vanishes ⇒ out == base(x).
    // (This is the just-loaded-before-training state; an inference adapter has
    // a trained, non-zero lora_b, but the math must reduce correctly.)
    let zero_b = Array::zeros::<f32>(&(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: zero_b,
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = LoRALinear::new(base, params, 20.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[1.0, 2.0], 1e-5);
  }

  // ───────────────────── fuse == forward ─────────────────────

  #[test]
  fn lora_fuse_matches_forward() {
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();

    let mut via_forward = layer.forward(&x).unwrap();
    // Fuse, then run the fused base's plain forward — must match.
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-5,
    );
  }

  #[test]
  fn lora_fuse_with_bias_matches_forward() {
    let bias = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap();
    let base = BaseLinear::dense(base_weight(), Some(bias)).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut via_forward = layer.forward(&x).unwrap();
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-5,
    );
  }

  // ───────────────────── DoRA forward ─────────────────────

  #[test]
  fn dora_linear_forward_hand_traced() {
    // DoRA with m chosen to equal ‖adapted‖₂ so the renorm is the identity,
    // making the expected output the same [3, 6] as the LoRA case — this
    // isolates the renorm wiring (m/denom == 1 row-wise).
    //
    // adapted = W + scale*(lora_bᵀ @ lora_aᵀ); with scale=2,
    //   lora_bᵀ = [[1,0],[0,1]], lora_aᵀ = [[1,0,0],[0,1,0]]
    //   lora_bᵀ @ lora_aᵀ = [[1,0,0],[0,1,0]]
    //   adapted = [[1,0,0],[0,1,0]] + 2*[[1,0,0],[0,1,0]] = [[3,0,0],[0,3,0]]
    //   ‖adapted‖₂ row-wise = [3, 3]
    // Set m = [3, 3] ⇒ m/denom = [1, 1] ⇒ out == LoRA out == [3, 6].
    let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
  }

  #[test]
  fn dora_linear_forward_renorm_halves() {
    // Same adapted norm [3, 3], but m = [1.5, 1.5] ⇒ m/denom = [0.5, 0.5] ⇒
    // out = 0.5 * [3, 6] = [1.5, 3.0].
    let m = Array::from_slice::<f32>(&[1.5, 1.5], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[1.5, 3.0], 1e-5);
  }

  #[test]
  fn dora_fuse_matches_forward() {
    let m = Array::from_slice::<f32>(&[1.5, 2.5], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut via_forward = layer.forward(&x).unwrap();
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-4,
    );
  }

  #[test]
  fn dora_requires_magnitude() {
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = DoRALinear::new(base, plain_params(), 2.0).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
  }

  // ───────────────────── QLoRA (quantized base) ─────────────────────

  #[test]
  fn qlora_forward_matches_dense_within_quant_error() {
    // Quantize a dense base, wrap with LoRA, and assert the QLoRA forward is
    // close to the dense LoRA forward (within affine-quant error). Use a
    // group_size that divides input_dims and a wide-ish weight so the quant
    // error stays small.
    //
    // input_dims must be divisible by group_size; use input_dims=64,
    // output_dims=2, group_size=32, bits=8 (low error).
    let input_dims = 64usize;
    let output_dims = 2usize;
    // Dense weight: row 0 = 1.0s, row 1 = 0.5s (well-represented at 8 bits).
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

    // lora_a [input_dims, r=2] small constant; lora_b [r=2, output_dims].
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: None,
    };

    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    // Dense LoRA forward.
    let dense_base = BaseLinear::dense(dense_w.try_clone().unwrap(), None).unwrap();
    let dense_layer = LoRALinear::new(dense_base, params.try_clone().unwrap(), 2.0).unwrap();
    let mut dense_out = dense_layer.forward(&x).unwrap();

    // Quantized base (affine, group_size=32, bits=8).
    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = LoRALinear::new(q_base, params, 2.0).unwrap();
    let mut q_out = q_layer.forward(&x).unwrap();

    // Within affine-quant error (8-bit, uniform weights → small).
    approx_eq(
      &q_out.to_vec::<f32>().unwrap(),
      &dense_out.to_vec::<f32>().unwrap(),
      1e-2,
    );
  }

  #[test]
  fn qlora_fuse_dequantize_matches_forward() {
    // fuse(dequantize=true) on a quantized base yields a dense fused linear
    // whose forward matches the QLoRA forward within quant error.
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: None,
    };
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = LoRALinear::new(q_base, params, 2.0).unwrap();
    let mut via_forward = q_layer.forward(&x).unwrap();

    let fused = q_layer.fuse(true).unwrap();
    assert!(matches!(fused, BaseLinear::Dense { .. }));
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-2,
    );
  }

  // ───────────────────── config parsing ─────────────────────

  #[test]
  fn config_parse_lora_basic() {
    let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 4,
      "lora_parameters": { "rank": 16, "scale": 20.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
    assert_eq!(cfg.num_layers, 4);
    assert_eq!(cfg.rank(), 16);
    assert_eq!(cfg.scale(), 20.0);
    assert!(!cfg.is_dora());
  }

  #[test]
  fn config_parse_peft_r_alias() {
    // A PEFT-style adapter_config.json names the rank `r` (not `rank`). The
    // `#[serde(alias = "r")]` must pick it up so a PEFT-trained adapter does
    // NOT silently fall back to the default rank.
    let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 4,
      "lora_parameters": { "r": 16, "lora_alpha": 32.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank(), 16, "PEFT `r` key must populate `rank`");
    // alpha/rank = 32/16 = 2.0 — the alias feeding `rank` makes the scale right.
    assert_eq!(cfg.scale(), 2.0);
  }

  #[test]
  fn config_parse_dora_and_alpha_scale() {
    // alpha/rank scale: alpha=32, rank=8 ⇒ scale=4.0. fine_tune_type dora.
    let json = r#"{
      "fine_tune_type": "dora",
      "num_layers": 2,
      "lora_parameters": { "rank": 8, "lora_alpha": 32.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(cfg.is_dora());
    assert_eq!(cfg.scale(), 4.0);
  }

  #[test]
  fn config_use_dora_flag() {
    let json = r#"{
      "fine_tune_type": "lora",
      "use_dora": true,
      "lora_parameters": { "rank": 8, "scale": 10.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(cfg.is_dora());
  }

  #[test]
  fn config_defaults_and_unknown_keys_ignored() {
    // Minimal config + extra training-only keys → parses, defaults applied.
    let json = r#"{ "optimizer": "adam", "learning_rate": 1e-4 }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
    assert_eq!(cfg.num_layers, DEFAULT_NUM_LAYERS);
    assert_eq!(cfg.rank(), DEFAULT_LORA_RANK);
    assert_eq!(cfg.scale(), DEFAULT_LORA_SCALE);
  }

  #[test]
  fn config_unknown_fine_tune_type_is_err() {
    let json = r#"{ "fine_tune_type": "bogus" }"#;
    assert!(LoraConfig::from_json(json).is_err());
  }

  // ───────────────────── path/key helpers ─────────────────────

  #[test]
  fn path_key_matching() {
    assert!(path_matches_key(
      "model.layers.27.self_attn.q_proj",
      "self_attn.q_proj"
    ));
    assert!(path_matches_key("self_attn.q_proj", "self_attn.q_proj"));
    assert!(!path_matches_key(
      "model.layers.27.self_attn.k_proj",
      "q_proj"
    ));
    // Must match on a segment boundary, not a substring.
    assert!(!path_matches_key("model.xq_proj", "q_proj"));
  }

  #[test]
  fn block_index_parsing() {
    assert_eq!(
      parse_block_index("model.layers.27.self_attn.q_proj"),
      Some(27)
    );
    assert_eq!(parse_block_index("model.layers.0.mlp.down_proj"), Some(0));
    assert_eq!(parse_block_index("model.embed_tokens"), None);
    assert_eq!(parse_block_index("lm_head"), None);
  }

  // ───────────────────── linear_to_lora_layers ─────────────────────

  /// Build a tiny weight map with 4 decoder blocks, each carrying a single
  /// `self_attn.q_proj.weight` (and one block also a `k_proj`), plus a
  /// top-level `lm_head.weight`.
  fn toy_weights() -> Weights {
    let mut w = Weights::new();
    for b in 0..4 {
      w.insert(
        format!("model.layers.{b}.self_attn.q_proj.weight"),
        base_weight(),
      );
    }
    w.insert(
      "model.layers.0.self_attn.k_proj.weight".to_string(),
      base_weight(),
    );
    w.insert("lm_head.weight".to_string(), base_weight());
    w
  }

  /// Adapter params for every q_proj path in the toy map (4 blocks).
  fn toy_adapter_params() -> HashMap<String, AdapterParams> {
    toy_adapter_params_for(&[0, 1, 2, 3])
  }

  /// Adapter params for the q_proj paths of the given block indices only.
  /// Used to keep an adapter's factor set aligned with the `num_layers` window
  /// under test — the completeness postcondition rejects factors for a path
  /// outside the selection, so a windowed test must supply only in-window
  /// factors.
  fn toy_adapter_params_for(blocks: &[i32]) -> HashMap<String, AdapterParams> {
    let mut m = HashMap::new();
    for &b in blocks {
      m.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    m
  }

  #[test]
  fn lora_layers_keys_and_num_layers_window() {
    // keys=["self_attn.q_proj"], num_layers=2 ⇒ only blocks 2,3's q_proj wrap.
    // The adapter supplies factors for exactly those two blocks (an adapter
    // that also carried block-0/1 factors would now be a config mismatch — see
    // `lora_layers_extra_factors_outside_window_is_err`).
    let weights = toy_weights();
    let params = toy_adapter_params_for(&[2, 3]);
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 2,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.q_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    // Only blocks 2 and 3 are inside the trailing-2 window.
    assert!(layers.contains_key("model.layers.2.self_attn.q_proj"));
    assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.1.self_attn.q_proj"));
    // k_proj never matches the key.
    assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
    // lm_head is a non-block path and not in keys → untouched.
    assert!(!layers.contains_key("lm_head"));
    assert_eq!(layers.len(), 2);
  }

  #[test]
  fn lora_layers_covers_all_blocks_when_num_layers_large() {
    let weights = toy_weights();
    let params = toy_adapter_params();
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 16, // > 4 blocks ⇒ all q_proj blocks wrap
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.q_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
  }

  // ───────────────────── load_adapters end-to-end ─────────────────────

  /// Write a mock adapter dir: adapter_config.json + adapters.safetensors with
  /// factors for two q_proj paths.
  fn write_mock_adapter(dir: &Path, fine_tune_type: &str, with_m: bool) {
    let config = format!(
      r#"{{
        "fine_tune_type": "{fine_tune_type}",
        "num_layers": 16,
        "lora_parameters": {{ "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }}
      }}"#
    );
    std::fs::write(dir.join("adapter_config.json"), config).unwrap();

    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(format!("{path}.lora_a"), lora_a());
      arrays.insert(format!("{path}.lora_b"), lora_b());
      if with_m {
        // m = ‖adapted‖₂ (so renorm is identity) → [3, 3] for these factors.
        arrays.insert(
          format!("{path}.m"),
          Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap(),
        );
      }
    }
    crate::io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
  }

  #[test]
  fn load_adapters_lora_end_to_end() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_lora_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "lora", false);

    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    // 4 q_proj blocks adapted.
    assert_eq!(layers.len(), 4);
    assert!(matches!(
      layers.get("model.layers.0.self_attn.q_proj"),
      Some(LoraLayer::Lora(_))
    ));

    // Forward through an adapted layer matches the hand-traced LoRA result.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layers
      .get("model.layers.0.self_attn.q_proj")
      .unwrap()
      .forward(&x)
      .unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);

    std::fs::remove_dir_all(&tmp).ok();
  }

  /// Write a mock adapter dir whose `adapter_config.json` is `config_json`
  /// (caller-supplied, so a test can vary `rank`/`r`/`alpha`) and whose
  /// `adapters.safetensors` carries rank-`r` factors for the 4 q_proj paths
  /// over the toy `[2, 3]` base: `lora_a` is `[3, r]`, `lora_b` is `[r, 2]`.
  fn write_mock_adapter_rank(dir: &Path, config_json: &str, r: usize) {
    std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();
    let la = Array::full::<f32>(&(3usize, r), 0.01).unwrap();
    let lb = Array::full::<f32>(&(r, 2usize), 0.01).unwrap();
    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(format!("{path}.lora_a"), la.try_clone().unwrap());
      arrays.insert(format!("{path}.lora_b"), lb.try_clone().unwrap());
    }
    crate::io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
  }

  #[test]
  fn load_adapters_peft_r_alias_rank16_loads() {
    // PEFT-style config: rank under the key `r` (not `rank`), `lora_alpha`
    // present. The `r` alias makes `config.rank() == 16`, so rank-16 factor
    // tensors pass the config-rank cross-check and load cleanly.
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_r16_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "r": 16, "lora_alpha": 32.0, "keys": ["self_attn.q_proj"] }
    }"#;
    write_mock_adapter_rank(&tmp, cfg, 16);
    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    // The resolved scale is alpha/rank = 32/16 = 2.0 — i.e. the `r` alias fed
    // the rank that divides alpha.
    if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
      assert_eq!(l.scale(), 2.0);
    } else {
      panic!("expected a LoRA layer");
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_rank_drift_is_shape_mismatch() {
    // Config declares rank 8 with `lora_alpha` present, but the factor tensors
    // are rank 16 (a stale config / unrecognized `r` drift). Without the
    // config-vs-tensor rank cross-check this silently builds rank-16 factors
    // and scales by alpha/8 instead of alpha/16 — wrong strength. It must now
    // fail loudly at load with a ShapeMismatch.
    let tmp = std::env::temp_dir().join(format!("mlxrs_rankdrift_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 8, "lora_alpha": 32.0, "keys": ["self_attn.q_proj"] }
    }"#;
    write_mock_adapter_rank(&tmp, cfg, 16);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::ShapeMismatch { .. }),
      "rank drift must be a ShapeMismatch, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_dora_end_to_end() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_dora_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "dora", true);

    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    assert!(matches!(
      layers.get("model.layers.0.self_attn.q_proj"),
      Some(LoraLayer::Dora(_))
    ));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_dora_missing_magnitude_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_dora_nom_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    // fine_tune_type dora but no `.m` arrays → recoverable Err.
    write_mock_adapter(&tmp, "dora", false);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_full_is_unsupported_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_full_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "full", false);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_unknown_fine_tune_type_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_bogus_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "bogus", false);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_missing_config_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_nocfg_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    // Only write the safetensors, no config.
    let arrays: HashMap<String, Array> = HashMap::new();
    crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_missing_dir_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_nodir_test_{}", std::process::id()));
    // Do NOT create the dir.
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
  }

  // ───────────────────── factor-shape validation ─────────────────────

  #[test]
  fn lora_rejects_mismatched_output_dims() {
    // lora_b last axis (3) != base output_dims (2).
    let bad_b = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: bad_b,
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = LoRALinear::new(base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
  }

  #[test]
  fn lora_rejects_rank_mismatch() {
    // lora_a [3, 2] but lora_b [3, 2] (leading 3 != a's r=2).
    let bad_b = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: bad_b,
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = LoRALinear::new(base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
  }

  // ───────── Finding 5: lora_a input-dim cross-check ─────────

  #[test]
  fn lora_rejects_wrong_lora_a_input_dim_dense() {
    // Dense base W is [output_dims=2, input_dims=3]; a lora_a with leading axis
    // 2 (≠ input_dims 3) must be rejected at construction, not deferred to a
    // mlx-c matmul failure on the first forward.
    let bad_a = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: bad_a,
      lora_b: lora_b(),
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = LoRALinear::new(base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
  }

  #[test]
  fn lora_rejects_wrong_lora_a_input_dim_quantized() {
    // Quantized base: dense [2, 64] affine-quantized at 8 bits ⇒ packed [2, 16];
    // base_input_dims recovers 16 * 32 / 8 = 64. A lora_a with leading axis 32
    // (≠ 64) must be rejected at construction.
    let input_dims = 64usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(2, input_dims)).unwrap();
    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();

    // input_dims should be 64 — supply a wrong-width lora_a [32, 2].
    let bad_a = Array::full::<f32>(&(32usize, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: bad_a,
      lora_b: lb,
      magnitude: None,
    };
    let err = LoRALinear::new(q_base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
  }

  #[test]
  fn lora_a_correct_input_dim_quantized_ok() {
    // The positive companion: a correctly-sized lora_a [64, 2] over the same
    // quantized base constructs cleanly (base_input_dims == 64 == lora_a[0]).
    let input_dims = 64usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(2, input_dims)).unwrap();
    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: None,
    };
    assert!(LoRALinear::new(q_base, params, 2.0).is_ok());
  }

  // ───────── Finding 4: scale precedence (alpha wins) ─────────

  #[test]
  fn resolved_scale_alpha_only() {
    // alpha present, no scale ⇒ alpha / rank.
    let p = LoraParameters {
      rank: 8,
      scale: None,
      alpha: Some(32.0),
      keys: None,
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 4.0);
  }

  #[test]
  fn resolved_scale_scale_only() {
    // scale present, no alpha ⇒ the literal scale.
    let p = LoraParameters {
      rank: 8,
      scale: Some(7.5),
      alpha: None,
      keys: None,
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 7.5);
  }

  #[test]
  fn resolved_scale_alpha_wins_over_scale() {
    // BOTH present ⇒ alpha / rank WINS over the literal scale (PEFT precedence).
    // alpha=64, rank=16 ⇒ 4.0, NOT the literal 99.0.
    let p = LoraParameters {
      rank: 16,
      scale: Some(99.0),
      alpha: Some(64.0),
      keys: None,
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 4.0);
  }

  #[test]
  fn resolved_scale_neither_is_default() {
    // Neither present ⇒ DEFAULT_LORA_SCALE.
    let p = LoraParameters {
      rank: 8,
      scale: None,
      alpha: None,
      keys: None,
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), DEFAULT_LORA_SCALE);
  }

  #[test]
  fn resolved_scale_alpha_with_nonpositive_rank_falls_back() {
    // Defensive floor: alpha present but rank <= 0 ⇒ `alpha / rank` is
    // undefined ⇒ fall through to the literal scale, then the default.
    let p = LoraParameters {
      rank: 0,
      scale: Some(5.0),
      alpha: Some(32.0),
      keys: None,
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 5.0);
    let p_no_scale = LoraParameters {
      rank: -1,
      scale: None,
      alpha: Some(32.0),
      keys: None,
      dropout: None,
    };
    assert_eq!(p_no_scale.resolved_scale(), DEFAULT_LORA_SCALE);
  }

  #[test]
  fn config_both_scale_and_alpha_alpha_wins() {
    // adapter_config.json carrying BOTH scale and lora_alpha ⇒ alpha wins.
    let json = r#"{
      "fine_tune_type": "lora",
      "lora_parameters": { "rank": 8, "scale": 50.0, "lora_alpha": 16.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.scale(), 2.0); // 16 / 8, not the literal 50.0
  }

  // ───────── Finding 1: num_layers <= 0 selects ALL blocks ─────────

  #[test]
  fn lora_layers_num_layers_negative_one_selects_all_blocks() {
    // mlx-lm `model.layers[-max(-1,0):]` == `layers[-0:]` == `layers[0:]` ⇒
    // num_layers: -1 adapts EVERY decoder block, not none.
    let weights = toy_weights();
    let params = toy_adapter_params(); // factors for all 4 q_proj blocks
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: -1,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.q_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4, "num_layers=-1 must adapt all 4 blocks");
    for b in 0..4 {
      assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
    }
  }

  #[test]
  fn lora_layers_num_layers_zero_selects_all_blocks() {
    // num_layers: 0 ⇒ `max(0,0)=0` ⇒ `layers[-0:]` == all blocks too.
    let weights = toy_weights();
    let params = toy_adapter_params();
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 0,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.q_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4, "num_layers=0 must adapt all 4 blocks");
  }

  // ───────── Finding 2: adapter-completeness postcondition ─────────

  #[test]
  fn lora_layers_explicit_key_missing_factors_is_err() {
    // keys=["self_attn.q_proj"], num_layers covers all 4 blocks, but the
    // adapter only supplies factors for blocks 0,1 ⇒ blocks 2,3 are selected
    // targets with no factors ⇒ Err (case a).
    let weights = toy_weights();
    let params = toy_adapter_params_for(&[0, 1]);
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 16,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.q_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(message.contains("missing factors"), "got: {message}");
        assert!(
          message.contains("model.layers.2.self_attn.q_proj"),
          "got: {message}"
        );
      }
      other => panic!("expected Backend, got {other:?}"),
    }
  }

  #[test]
  fn lora_layers_unused_adapter_factor_is_err() {
    // The adapter carries a factor group for a path that exists in NO base
    // weight (a path-prefix mismatch / config drift) ⇒ Err (case b).
    let weights = toy_weights();
    let mut params = toy_adapter_params(); // all 4 q_proj blocks (all match)
    params.insert(
      "model.layers.99.self_attn.q_proj".to_string(),
      plain_params(),
    );
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 16,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.q_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(message.contains("match no base layer"), "got: {message}");
        assert!(
          message.contains("model.layers.99.self_attn.q_proj"),
          "got: {message}"
        );
      }
      other => panic!("expected Backend, got {other:?}"),
    }
  }

  #[test]
  fn lora_layers_empty_result_is_err() {
    // keys names a projection that exists in NO base weight, and there are no
    // factors ⇒ nothing adapted ⇒ Err (case c).
    let weights = toy_weights();
    let params: HashMap<String, AdapterParams> = HashMap::new();
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 16,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: Some(vec!["self_attn.nonexistent_proj".to_string()]),
        dropout: None,
      },
      use_dora: false,
    };
    let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("no base layer was adapted"),
          "got: {message}"
        );
      }
      other => panic!("expected Backend, got {other:?}"),
    }
  }

  #[test]
  fn lora_layers_autodiscovery_partial_factors_is_ok() {
    // keys: None (auto-discovery) ⇒ a base linear without factors is EXPECTED
    // (the adapter trains only a subset); only the unused-factor (b) and
    // empty-result (c) checks apply. Factors for 2 of the 4 q_proj blocks ⇒ Ok.
    let weights = toy_weights();
    let params = toy_adapter_params_for(&[2, 3]);
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      num_layers: 16,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: None,
        dropout: None,
      },
      use_dora: false,
    };
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 2);
  }

  #[test]
  fn load_adapters_unused_factor_end_to_end_is_err() {
    // End-to-end: an adapters.safetensors carrying a factor group for a path
    // absent from the base model ⇒ load_adapters rejects it.
    let tmp = std::env::temp_dir().join(format!("mlxrs_unused_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(format!("{path}.lora_a"), lora_a());
      arrays.insert(format!("{path}.lora_b"), lora_b());
    }
    // A factor group for a path that is NOT in toy_weights().
    arrays.insert(
      "model.layers.42.self_attn.q_proj.lora_a".to_string(),
      lora_a(),
    );
    arrays.insert(
      "model.layers.42.self_attn.q_proj.lora_b".to_string(),
      lora_b(),
    );
    crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_empty_safetensors_is_err() {
    // An empty adapters.safetensors (no factor groups at all) ⇒ nothing adapted
    // ⇒ Err (case c), instead of a silently-unadapted Ok.
    let tmp = std::env::temp_dir().join(format!("mlxrs_emptyst_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    let arrays: HashMap<String, Array> = HashMap::new();
    crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(matches!(err, Error::Backend { .. }));
    std::fs::remove_dir_all(&tmp).ok();
  }

  // ───────── Finding 3: QDoRA forward via quantized_matmul ─────────

  #[test]
  fn qdora_forward_matches_dense_within_quant_error() {
    // QDoRA (DoRA over a quantized base) + bias: the forward must match the
    // dense DoRA forward within affine-quant error. By construction the
    // quantized base output runs through quantized_matmul (base_output_no_bias),
    // never a full dense-weight matmul — the dequantized weight is materialized
    // only for the adapted-weight L2-norm.
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    // m = ‖adapted‖₂ row-wise of the DENSE adapted weight (so dense + quantized
    // share the same magnitude vector — the renorm is identical).
    let bias = Array::from_slice::<f32>(&[3.0, -1.0], &(output_dims,)).unwrap();

    let dense_params = AdapterParams {
      lora_a: la.try_clone().unwrap(),
      lora_b: lb.try_clone().unwrap(),
      magnitude: None,
    };
    // Build a DoRALinear over the dense base to read back its computed adapted
    // norm via fuse? Simpler: pick m = norm of (dense_w + scale*delta).
    let scale = 2.0f32;
    let delta = lora_delta(&dense_params, scale).unwrap();
    let adapted = dense_w.add(&delta).unwrap();
    let m = ops::linalg_full::norm(&adapted, 2.0, &[1], false).unwrap();

    let dense_base = BaseLinear::dense(
      dense_w.try_clone().unwrap(),
      Some(bias.try_clone().unwrap()),
    )
    .unwrap();
    let dense_layer = DoRALinear::new(
      dense_base,
      AdapterParams {
        lora_a: la.try_clone().unwrap(),
        lora_b: lb.try_clone().unwrap(),
        magnitude: Some(m.try_clone().unwrap()),
      },
      scale,
    )
    .unwrap();
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();
    let mut dense_out = dense_layer.forward(&x).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base = BaseLinear::quantized(
      w_q,
      scales,
      biases,
      Some(bias.try_clone().unwrap()),
      32,
      8,
      "affine".to_string(),
    )
    .unwrap();
    let q_layer = DoRALinear::new(
      q_base,
      AdapterParams {
        lora_a: la,
        lora_b: lb,
        magnitude: Some(m),
      },
      scale,
    )
    .unwrap();
    let mut q_out = q_layer.forward(&x).unwrap();

    approx_eq(
      &q_out.to_vec::<f32>().unwrap(),
      &dense_out.to_vec::<f32>().unwrap(),
      2e-2,
    );
  }

  #[test]
  fn qdora_forward_matches_fuse() {
    // QDoRA forward must equal its own fuse path within quant error — exercises
    // the quantized_matmul base output against the fused (renormalized) weight.
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let m = Array::from_slice::<f32>(&[1.5, 2.5], &(output_dims,)).unwrap();
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = DoRALinear::new(
      q_base,
      AdapterParams {
        lora_a: la,
        lora_b: lb,
        magnitude: Some(m),
      },
      2.0,
    )
    .unwrap();
    let mut via_forward = q_layer.forward(&x).unwrap();
    let fused = q_layer.fuse(true).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      2e-2,
    );
  }

  // ───────── Finding 6: adapters.safetensors hardening ─────────

  #[test]
  fn load_adapters_non_regular_safetensors_is_err() {
    // A directory planted where adapters.safetensors should be is not a regular
    // file ⇒ load_adapters rejects it before handing the path to mlx-c.
    let tmp = std::env::temp_dir().join(format!("mlxrs_nonreg_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    // adapters.safetensors is a DIRECTORY, not a file.
    std::fs::create_dir_all(tmp.join("adapters.safetensors")).unwrap();

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(message.contains("not a regular file"), "got: {message}");
      }
      other => panic!("expected Backend, got {other:?}"),
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_oversized_safetensors_is_err() {
    // A sparse file reporting a length beyond MAX_ADAPTER_SAFETENSORS_BYTES is
    // rejected on the stat, before any mmap. set_len makes a sparse file on
    // APFS/most filesystems — the on-disk footprint stays ~0.
    let tmp = std::env::temp_dir().join(format!("mlxrs_oversize_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    let f = std::fs::File::create(tmp.join("adapters.safetensors")).unwrap();
    f.set_len(MAX_ADAPTER_SAFETENSORS_BYTES + 1).unwrap();
    drop(f);

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(message.contains("adapter budget"), "got: {message}");
      }
      other => panic!("expected Backend, got {other:?}"),
    }
    std::fs::remove_dir_all(&tmp).ok();
  }
}
