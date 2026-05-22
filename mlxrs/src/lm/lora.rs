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
//! `scale` (mlx-lm's `scale = alpha / r` when built from `alpha`, else the
//! literal `scale` field — default `20.0`):
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

use std::{collections::HashMap, path::Path};

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
/// `alpha` is present instead (the PEFT/HF convention `scale = alpha / rank`),
/// [`LoraParameters::resolved_scale`] derives it. `keys` is the explicit
/// target-projection allowlist (e.g. `["self_attn.q_proj",
/// "self_attn.v_proj"]`); `None` means "every eligible linear" (mlx-lm's
/// auto-discovery, `tuner/utils.py:85-101`). `dropout` is carried for config
/// round-trip fidelity but **ignored at inference** (an inference adapter's
/// dropout is the identity — see the [module docs](self)).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LoraParameters {
  /// Low-rank dimension `r` (mlx-lm `config["rank"]`). Defaults to
  /// [`DEFAULT_LORA_RANK`].
  #[serde(default = "default_rank")]
  pub rank: i32,
  /// Literal low-rank scale (mlx-lm `config["scale"]`). Defaults to
  /// [`DEFAULT_LORA_SCALE`] when neither `scale` nor `alpha` is present.
  #[serde(default)]
  pub scale: Option<f32>,
  /// PEFT/HF `lora_alpha` — if present (and `scale` is not), the effective
  /// scale is `alpha / rank`. Carried so adapters trained with the HF
  /// convention load with the correct scale.
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
  /// The effective low-rank scale, resolving the mlx-lm / PEFT precedence:
  /// an explicit `scale` wins; else `alpha / rank` (the HF `lora_alpha`
  /// convention); else [`DEFAULT_LORA_SCALE`]. A non-positive `rank` with an
  /// `alpha` present cannot form `alpha / rank`, so it falls back to the
  /// default scale (the [`LoraConfig`] validator rejects `rank <= 0` before a
  /// layer is ever built, so this is a defensive floor, not a live path).
  pub fn resolved_scale(&self) -> f32 {
    if let Some(s) = self.scale {
      s
    } else if let Some(a) = self.alpha {
      if self.rank > 0 {
        a / self.rank as f32
      } else {
        DEFAULT_LORA_SCALE
      }
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
  /// Defaults to [`DEFAULT_NUM_LAYERS`]. A negative value is treated as `0`
  /// (no layers) — mlx-lm's `model.layers[-max(num_layers, 0):]`
  /// (`tuner/utils.py:103`).
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

  /// The base linear's output `y = x @ Wᵀ (+ bias)` — a plain matmul for the
  /// dense base, a fused [`ops::quantized::quantized_matmul`] (`transpose=true`)
  /// for the quantized base. Mirrors mlx-lm `self.linear(x)`
  /// (`tuner/lora.py:96`) / swift `super.callAsFunction(x)`. Does NOT add the
  /// low-rank term — that is [`LoRALinear::forward`]'s job.
  fn base_output(&self, x: &Array) -> Result<Array> {
    match self {
      BaseLinear::Dense { weight, bias } => {
        let wt = weight.transpose()?;
        let y = x.matmul(&wt)?;
        match bias {
          Some(b) => y.add(b),
          None => Ok(y),
        }
      }
      BaseLinear::Quantized {
        weight,
        scales,
        quant_biases,
        bias,
        group_size,
        bits,
        mode,
      } => {
        // `transpose=true` matches mlx-lm's QuantizedLinear (the packed weight
        // is laid out for the `output_dims x input_dims` orientation).
        let y = ops::quantized::quantized_matmul(
          x,
          weight,
          scales,
          quant_biases.as_ref(),
          true,
          *group_size,
          *bits,
          mode,
        )?;
        match bias {
          Some(b) => y.add(b),
          None => Ok(y),
        }
      }
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
/// base output and the adapted-weight norm both run against the dequantized
/// weight (mlx-lm `tuner/dora.py:92-106,113`).
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
  /// w       = dequantized_weight
  /// y       = x @ wᵀ
  /// z       = (x @ lora_a) @ lora_b
  /// out     = y + (scale · z)
  /// adapted = w + (scale · lora_bᵀ) @ lora_aᵀ
  /// denom   = ‖adapted‖₂ (axis 1)
  /// out     = (m / denom) · out  (+ bias)
  /// ```
  ///
  /// The renormalization `(m / denom)` is the weight-decomposition step that
  /// distinguishes DoRA from LoRA. Lazy — does not evaluate.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let w = self.base.dequantized_weight()?;
    // y = x @ wᵀ — DoRA computes the base output WITHOUT the base bias here;
    // the bias is re-added at the very end (mlx-lm `tuner/dora.py:113,126-127`),
    // AFTER the magnitude renormalization, so the renorm does not scale it.
    let wt = w.transpose()?;
    let y = x.matmul(&wt)?;

    let z = lora_z(x, &self.params)?;
    let scaled_z = scaled(&z, self.scale)?;
    let scaled_z = match x.dtype() {
      Ok(dt) => scaled_z.astype(dt)?,
      Err(_) => scaled_z,
    };
    let out = y.add(&scaled_z)?;

    // adapted = w + (scale · lora_bᵀ) @ lora_aᵀ; denom = ‖adapted‖₂ (axis 1).
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

/// Validate `lora_a` / `lora_b` against the base dims. `lora_b` is
/// `[r, output_dims]`, so its last axis must equal the base `output_dims`;
/// `lora_a` is `[input_dims, r]`, so its rank-2 shape's last axis (`r`) must
/// match `lora_b`'s leading axis (`r`). The `input_dims` axis of `lora_a` is
/// not cross-checked against the (packed) quantized base weight, whose last
/// axis is `input_dims * bits / 32`; mlx-c validates the matmul contract at
/// the forward call. Surfaces a recoverable [`Error::ShapeMismatch`].
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
/// - **`num_layers`** — only the **last** `max(num_layers, 0)` decoder blocks
///   are adapted (mlx-lm `model.layers[-max(num_layers, 0):]`,
///   `tuner/utils.py:103`). A path's block index is parsed from the
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
/// `adapters.safetensors`); a selected path with no entry in `adapter_params`
/// is skipped (no factors to apply). The total number of decoder blocks
/// (`num_blocks`) is needed to resolve the trailing-`num_layers` window; pass
/// the model's layer count (mlx-lm reads `len(model.layers)`).
///
/// A selected path whose factor shapes don't match the base (or a DoRA path
/// with no magnitude) is a recoverable [`Error::ShapeMismatch`] /
/// [`Error::Backend`].
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
  let num_layers = config.num_layers.max(0);
  // The first adapted block index: blocks [num_blocks - num_layers, num_blocks).
  let first_adapted = (num_blocks - num_layers).max(0);

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

    // Only build a layer for a path we actually have factors for.
    let Some(params) = adapter_params.get(path) else {
      continue;
    };

    let base = build_base_linear(weights, path, weight, quant)?;
    let layer = if is_dora {
      LoraLayer::Dora(DoRALinear::new(base, params.try_clone()?, scale)?)
    } else {
      LoraLayer::Lora(LoRALinear::new(base, params.try_clone()?, scale)?)
    };
    out.insert(path.to_string(), layer);
  }

  Ok(out)
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
/// - `fine_tune_type: "full"` (a full-weight fine-tune, not an adapter — see
///   [`FineTuneType::Full`]) → [`Error::Backend`] (unsupported here).
///   An **unknown** `fine_tune_type` string is a serde parse error →
///   [`Error::Backend`] from [`LoraConfig::from_json`].
/// - A target path with a magnitude-less DoRA factor, or factor shapes that
///   don't match the base → [`Error::ShapeMismatch`] / [`Error::Backend`].
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

  // 2) adapters.safetensors → per-path AdapterParams.
  let st_path = dir.join("adapters.safetensors");
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
    let mut m = HashMap::new();
    for b in 0..4 {
      m.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    m
  }

  #[test]
  fn lora_layers_keys_and_num_layers_window() {
    // keys=["self_attn.q_proj"], num_layers=2 ⇒ only blocks 2,3's q_proj wrap.
    let weights = toy_weights();
    let params = toy_adapter_params();
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
}
