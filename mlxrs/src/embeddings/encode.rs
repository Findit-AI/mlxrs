//! The `encode` entry — tokenize a batch of texts, pad to the batch's max
//! length, run an [`EmbeddingModel`], pool, and optionally L2-normalize into a
//! `(batch, dim)` embedding matrix.
//!
//! Ports the orchestration of:
//! - python `mlx-embeddings` `utils.py::generate` (tokenize via the processor
//!   with `padding` / `truncation` / `max_length`, run the model, return the
//!   embeddings) cross-referenced with `models/pooling.py::pool_by_config` and
//!   `models/base.py::normalize_embeddings`;
//! - swift `MLXEmbedders` `EmbedderModelContainer.perform` (encode each text →
//!   pad to the batch max → build the mask → `model(padded, …, attentionMask:
//!   mask)` → `pooling(output, normalize: …)` → `eval`).
//!
//! Unlike python, where the per-architecture model returns an already pooled +
//! normalized `text_embeds`, mlxrs pools *externally* with the existing
//! [`pool`] dispatcher (the no-model-arch rule keeps per-model heads out of
//! scope), exactly as swift's container does. Tokenization is local-only via
//! the existing [`Tokenizer`]; pooling and normalization reuse
//! [`crate::embeddings::pool`] — nothing here re-implements them.

use crate::{
  array::Array,
  error::{Error, Result, try_with_capacity},
  tokenizer::Tokenizer,
};

use super::{PoolingStrategy, model::EmbeddingModel, pool};

/// Configuration for [`encode`].
///
/// Defaults mirror python `generate` (`max_length = 512`, padding +
/// truncation on, special tokens added) composed with swift's
/// `pooling(output, normalize: true)`: [`mean`](PoolingStrategy::Mean)
/// pooling, L2-normalized output.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
  /// Pooling strategy applied to the model's `(batch, seq_len, hidden)`
  /// hidden states (the existing [`PoolingStrategy`] / [`pool`] dispatcher).
  /// Default [`PoolingStrategy::Mean`] (python `generate`'s `text_embeds` is
  /// "mean pooled and normalized"; swift container default).
  pub strategy: PoolingStrategy,
  /// L2-normalize the pooled vectors (python `normalize_embeddings`, swift
  /// `pooling(_, normalize: true)`). Default `true`.
  pub normalize: bool,
  /// Add the tokenizer's special tokens (BOS/EOS/sep) when encoding, as in
  /// python `processor(..., add_special_tokens=True)` (the transformers
  /// default) and swift `tokenizer.encode(text:, addSpecialTokens: true)`.
  /// Default `true`.
  pub add_special_tokens: bool,
  /// Per-sequence hard token cap (python `truncation=True`,
  /// `max_length=512`): each text is right-truncated (keep the head, drop the
  /// tail) to at most this many ids *before* batch padding. `None` disables
  /// truncation. Default `Some(512)`.
  pub max_length: Option<usize>,
  /// Token id written into padding positions. The attention mask is `0`
  /// there, so this value never reaches the pooled output — it exists only so
  /// the padded `(batch, seq_len)` id tensor is well-formed (swift pads with
  /// `0`). Default `0`.
  pub pad_token_id: u32,
  /// Optional matryoshka last-dim truncation forwarded to [`pool`] (swift
  /// `Pooling.dimension`). `None` keeps the model's full hidden width.
  /// Default `None`.
  pub dimension: Option<usize>,
  /// Apply a fused LayerNorm to the pooled vector before truncation /
  /// normalization (swift `applyLayerNorm:`), forwarded to [`pool`]. Default
  /// `false`.
  pub apply_layer_norm: bool,
  /// Apply a fused RMSNorm to the pooled vector (mlx-c-surfaced variant;
  /// ignored if `apply_layer_norm` is also set), forwarded to [`pool`].
  /// Default `false`.
  pub apply_rms_norm: bool,
}

impl Default for EncodeConfig {
  fn default() -> Self {
    Self {
      strategy: PoolingStrategy::Mean,
      normalize: true,
      add_special_tokens: true,
      max_length: Some(512),
      pad_token_id: 0,
      dimension: None,
      apply_layer_norm: false,
      apply_rms_norm: false,
    }
  }
}

impl EncodeConfig {
  /// Fluent builder constructor; equivalent to [`EncodeConfig::default`].
  pub fn new() -> Self {
    Self::default()
  }
  /// Set [`EncodeConfig::strategy`].
  pub fn strategy(mut self, v: PoolingStrategy) -> Self {
    self.strategy = v;
    self
  }
  /// Set [`EncodeConfig::normalize`].
  pub fn normalize(mut self, v: bool) -> Self {
    self.normalize = v;
    self
  }
  /// Set [`EncodeConfig::add_special_tokens`].
  pub fn add_special_tokens(mut self, v: bool) -> Self {
    self.add_special_tokens = v;
    self
  }
  /// Set [`EncodeConfig::max_length`].
  pub fn max_length(mut self, v: Option<usize>) -> Self {
    self.max_length = v;
    self
  }
  /// Set [`EncodeConfig::pad_token_id`].
  pub fn pad_token_id(mut self, v: u32) -> Self {
    self.pad_token_id = v;
    self
  }
  /// Set [`EncodeConfig::dimension`].
  pub fn dimension(mut self, v: Option<usize>) -> Self {
    self.dimension = v;
    self
  }
  /// Set [`EncodeConfig::apply_layer_norm`].
  pub fn apply_layer_norm(mut self, v: bool) -> Self {
    self.apply_layer_norm = v;
    self
  }
  /// Set [`EncodeConfig::apply_rms_norm`].
  pub fn apply_rms_norm(mut self, v: bool) -> Self {
    self.apply_rms_norm = v;
    self
  }
}

/// Tokenize `texts`, right-pad each id row to the batch's max length with
/// `pad_token_id`, and build the matching `(batch, seq_len)` attention mask
/// (`1` for real tokens, `0` for padding).
///
/// Returns `(input_ids, attention_mask, seq_len)`:
/// - `input_ids` — `(batch, seq_len)` `u32` array (right-padded);
/// - `attention_mask` — `(batch, seq_len)` `f32` array (`1.0` / `0.0`);
/// - `seq_len` — the batch max length (after per-text truncation).
///
/// `seq_len` is the longest *truncated* row, so it never exceeds
/// `max_length`. An empty `texts` slice, or a batch whose every row is empty
/// (e.g. `max_length = Some(0)`), produces `seq_len = 0` and correspondingly
/// shaped `(batch, 0)` arrays (an all-padding batch — the mask is all-`0`,
/// which the mean / max poolers floor / guard).
///
/// Right-padding (and the resulting trailing-`0` mask) matches the HF
/// tokenizer's default `padding_side="right"` for encoders and swift's
/// container, so the existing mask-aware poolers behave as in the references.
fn tokenize_and_pad(
  tokenizer: &Tokenizer,
  texts: &[&str],
  add_special_tokens: bool,
  max_length: Option<usize>,
  pad_token_id: u32,
) -> Result<(Array, Array, usize)> {
  let batch = texts.len();

  // Tokenize each text, applying the per-sequence truncation cap.
  let mut rows: Vec<Vec<u32>> = try_with_capacity(batch)?;
  for &text in texts {
    let mut ids = tokenizer.encode(text, add_special_tokens)?;
    if let Some(cap) = max_length
      && ids.len() > cap
    {
      ids.truncate(cap);
    }
    rows.push(ids);
  }

  let seq_len = rows.iter().map(Vec::len).max().unwrap_or(0);

  // Flatten into right-padded (batch, seq_len) id + mask buffers.
  let total = batch
    .checked_mul(seq_len)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("encode: batch {batch} * seq_len {seq_len} overflows usize"),
    })?;
  let mut id_data: Vec<u32> = try_with_capacity(total)?;
  let mut mask_data: Vec<f32> = try_with_capacity(total)?;
  for row in &rows {
    let real = row.len();
    id_data.extend_from_slice(row);
    mask_data.extend(std::iter::repeat_n(1.0_f32, real));
    let pad = seq_len - real;
    id_data.extend(std::iter::repeat_n(pad_token_id, pad));
    mask_data.extend(std::iter::repeat_n(0.0_f32, pad));
  }

  let input_ids = Array::from_slice::<u32>(&id_data, &(batch, seq_len))?;
  let attention_mask = Array::from_slice::<f32>(&mask_data, &(batch, seq_len))?;
  Ok((input_ids, attention_mask, seq_len))
}

/// Encode a batch of texts into a `(batch, dim)` embedding matrix.
///
/// Pipeline (python `generate` ∘ swift `EmbedderModelContainer.perform`):
/// 1. tokenize each text (special tokens per `cfg.add_special_tokens`),
///    right-truncate to `cfg.max_length`;
/// 2. right-pad every id row to the batch's max length and build the matching
///    `(batch, seq_len)` attention mask (`1` real, `0` pad);
/// 3. run `model.forward(input_ids, attention_mask)` → hidden states;
/// 4. pool with `cfg.strategy` and apply `cfg.{apply_layer_norm,
///    apply_rms_norm, dimension, normalize}` via the existing [`pool`]
///    dispatcher.
///
/// The returned array is `(batch, dim)` (or `(batch, seq_len, dim)` if
/// `cfg.strategy` is [`PoolingStrategy::None`], which passes the hidden states
/// through). **No implicit eval**: the result is a lazy graph node; the caller
/// evaluates (or reads it) when ready.
///
/// An empty `texts` slice returns a `(0, …)` array (zero-row batch). The
/// pooling stage receives the model's hidden states unchanged from the
/// reference behavior — mask-aware poolers exclude the padded tail.
///
/// - `model` — any [`EmbeddingModel`] (trait object: one call site, many
///   architectures);
/// - `tokenizer` — the loaded [`Tokenizer`] (local-only; no network);
/// - `texts` — the batch to encode;
/// - `cfg` — pooling / normalization / tokenization knobs ([`EncodeConfig`]).
pub fn encode(
  model: &dyn EmbeddingModel,
  tokenizer: &Tokenizer,
  texts: &[&str],
  cfg: &EncodeConfig,
) -> Result<Array> {
  let (input_ids, attention_mask, _seq_len) = tokenize_and_pad(
    tokenizer,
    texts,
    cfg.add_special_tokens,
    cfg.max_length,
    cfg.pad_token_id,
  )?;

  let output = model.forward(&input_ids, &attention_mask)?;

  pool(
    &output.last_hidden_state,
    &attention_mask,
    cfg.strategy,
    cfg.normalize,
    cfg.dimension,
    cfg.apply_layer_norm,
    cfg.apply_rms_norm,
  )
}

#[cfg(test)]
mod tests {
  //! Hand-traced `encode` tests over a [`MockEmbeddingModel`]: a real
  //! tokenizer encodes a 2-text batch, the padding / mask logic is asserted
  //! explicitly, and mean / cls pooling + L2-normalization are checked against
  //! values computed by hand from the canned hidden states.

  use super::*;
  use crate::embeddings::model::MockEmbeddingModel;

  const TOL: f32 = 1e-5;

  fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL
  }

  fn vclose(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
  }

  /// A whitespace word-level tokenizer with no special tokens: each distinct
  /// word maps to a stable id (`a`→0 … `e`→4). Built in-memory via the public
  /// `tokenizers` API, serialized to a temp `tokenizer.json`, and loaded
  /// through [`Tokenizer::from_path`] — the same feature-combo-agnostic load
  /// path the integration tests use (no dependence on the cfg-gated
  /// `from_loaded` signature). Two texts of different word counts exercise the
  /// pad / mask path.
  fn word_tokenizer() -> Tokenizer {
    use tokenizers::{
      Tokenizer as HfTokenizer, models::wordlevel::WordLevel,
      pre_tokenizers::whitespace::Whitespace,
    };

    // `WordLevelBuilder::vocab` takes the crate's `AHashMap<String, u32>`;
    // collect into it via the arg's inferred type (no extra dep named).
    let vocab = ["a", "b", "c", "d", "e"]
      .iter()
      .enumerate()
      .map(|(i, w)| ((*w).to_string(), i as u32))
      .collect();
    let wl = WordLevel::builder()
      .vocab(vocab)
      .unk_token("a".to_string())
      .build()
      .unwrap();
    let mut hf = HfTokenizer::new(wl);
    hf.with_pre_tokenizer(Some(Whitespace {}));

    // Serialize to a per-process temp dir (write-once), then load via the
    // public `from_path`. The content is deterministic; a `OnceLock` removes
    // the parallel-test write race while every test reads the same file.
    static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    let dir = FIXTURE.get_or_init(|| {
      let dir = std::env::temp_dir().join(format!("mlxrs-emb-encode-tok-{}", std::process::id()));
      std::fs::create_dir_all(&dir).unwrap();
      hf.save(dir.join("tokenizer.json"), false).unwrap();
      dir
    });
    Tokenizer::from_path(dir, None).unwrap()
  }

  #[test]
  fn tokenize_and_pad_builds_right_padded_ids_and_mask() {
    let tok = word_tokenizer();
    // "a b c" -> [0,1,2] ; "d e" -> [3,4]. Batch max len = 3.
    let (mut ids, mut mask, seq_len) =
      tokenize_and_pad(&tok, &["a b c", "d e"], false, None, 7).unwrap();
    assert_eq!(seq_len, 3);
    assert_eq!(ids.shape(), vec![2, 3]);
    assert_eq!(mask.shape(), vec![2, 3]);
    // Row 1 is right-padded with pad_token_id = 7 and mask 0 in the tail.
    assert_eq!(ids.to_vec::<u32>().unwrap(), vec![0, 1, 2, 3, 4, 7]);
    assert_eq!(
      mask.to_vec::<f32>().unwrap(),
      vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    );
  }

  #[test]
  fn tokenize_and_pad_truncates_to_max_length() {
    let tok = word_tokenizer();
    // max_length = 2 trims "a b c" to [0,1]; "d e" is already 2. seq_len = 2.
    let (mut ids, mut mask, seq_len) =
      tokenize_and_pad(&tok, &["a b c", "d e"], false, Some(2), 0).unwrap();
    assert_eq!(seq_len, 2);
    assert_eq!(ids.to_vec::<u32>().unwrap(), vec![0, 1, 3, 4]);
    assert_eq!(mask.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn encode_mean_pool_normalized_two_text_batch() {
    let tok = word_tokenizer();
    // Canned per-position hidden rows (hidden = 2):
    //   pos0 = [1, 0], pos1 = [0, 1], pos2 = [1, 1]
    let model = MockEmbeddingModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);

    // "a b c" -> 3 real tokens (pos0,1,2); "d e" -> 2 real (pos0,1) + 1 pad.
    let cfg = EncodeConfig::new()
      .add_special_tokens(false)
      .strategy(PoolingStrategy::Mean)
      .normalize(true);
    let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
    assert_eq!(emb.shape(), vec![2, 2]);
    let v = emb.to_vec::<f32>().unwrap();

    // Row 0 mean over pos0,1,2 = ([1,0]+[0,1]+[1,1])/3 = [2/3, 2/3];
    // L2-normalized = [1/√2, 1/√2].
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    assert!(vclose(&v[0..2], &[inv_sqrt2, inv_sqrt2]));

    // Row 1 mean over pos0,1 (pad excluded by mask) = ([1,0]+[0,1])/2 = [0.5,0.5];
    // L2-normalized = [1/√2, 1/√2].
    assert!(vclose(&v[2..4], &[inv_sqrt2, inv_sqrt2]));
  }

  #[test]
  fn encode_mean_pool_unnormalized_excludes_padding() {
    let tok = word_tokenizer();
    let model = MockEmbeddingModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
    let cfg = EncodeConfig::new()
      .add_special_tokens(false)
      .strategy(PoolingStrategy::Mean)
      .normalize(false);
    let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
    let v = emb.to_vec::<f32>().unwrap();
    // Row 0: [2/3, 2/3] ; Row 1: [0.5, 0.5] (pad position excluded).
    assert!(vclose(&v[0..2], &[2.0 / 3.0, 2.0 / 3.0]));
    assert!(vclose(&v[2..4], &[0.5, 0.5]));
  }

  #[test]
  fn encode_cls_pool_selects_first_real_token() {
    let tok = word_tokenizer();
    // pos0 distinctive so CLS (first real token) is identifiable.
    let model = MockEmbeddingModel::new(vec![vec![9.0, 3.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
    let cfg = EncodeConfig::new()
      .add_special_tokens(false)
      .strategy(PoolingStrategy::Cls)
      .normalize(false);
    let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
    assert_eq!(emb.shape(), vec![2, 2]);
    let v = emb.to_vec::<f32>().unwrap();
    // Both rows are right-padded, so the first real token is pos0 = [9, 3].
    assert!(vclose(&v[0..2], &[9.0, 3.0]));
    assert!(vclose(&v[2..4], &[9.0, 3.0]));
  }
}
