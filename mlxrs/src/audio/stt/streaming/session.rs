//! Streaming inference session — orchestrates
//! [`super::mel_spectrogram::IncrementalMelSpectrogram`] +
//! [`super::encoder::StreamingEncoder`] + a per-architecture decoder to
//! produce a [`super::types::TranscriptionEvent`] stream.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/StreamingInferenceSession.swift`][swift-ref]
//! adapted to mlxrs's synchronous foreground-only execution model:
//!
//! - The Swift reference launches `Task.detached { ... runDecodePass
//!   ... }` per pass and yields events into an
//!   `AsyncStream<TranscriptionEvent>`. mlxrs runs each decode pass
//!   synchronously on the caller's thread; events are returned as a
//!   batch (`Vec<TranscriptionEvent>`) from
//!   [`StreamingInferenceSession::feed_audio`] and
//!   [`StreamingInferenceSession::stop`].
//! - The Swift reference depends on the concrete `Qwen3ASRModel`
//!   (`audioTower`, `tokenizer`, `mergeAudioFeatures`, `buildPrompt`,
//!   `makeCache`, `callAsFunction`). mlxrs replaces that with the
//!   [`StreamingDecoderBackend`] trait every per-architecture model
//!   implements — same orchestration loop, no concrete model in the
//!   port (per the [no per-model arch porting][noarch] rule).
//! - The Swift session uses Apple's `OSAllocatedUnfairLock` + tokenizer
//!   protocol. mlxrs uses owned `&mut self` (single-threaded session) +
//!   a [`StreamingTokenizer`] trait the caller supplies.
//!
//! The promotion / agreement / boundary-boost logic mirrors the Swift
//! reference at-line: a token is promoted to confirmed when it has been
//! seen for `>= min_agreement_passes` consecutive decode passes AND has
//! survived for `>= delay_preset.delay_ms()`. When a full encoder window
//! completes (or
//! [`super::types::StreamingConfig::finalize_completed_windows`] is on
//! and the boundary fast cadence elapses) the session promotes the
//! current provisional run, finalizes the window's text, and resets
//! decode state for the next window.
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/StreamingInferenceSession.swift
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

use std::{collections::VecDeque, time::Instant};

use super::{
  encoder::{StreamingEncoder, StreamingEncoderBackend},
  mel_spectrogram::IncrementalMelSpectrogram,
  types::{StreamingConfig, StreamingStats, TranscriptionEvent},
};
use crate::{Array, error::Result};

/// Architecture-specific per-pass decoder bridge.
///
/// Implementors wrap the per-model audio-decoder forward pass (the
/// Swift reference's `buildPrompt` + `mergeAudioFeatures` + KV-cache +
/// auto-regressive sampling loop). The session calls
/// [`StreamingDecoderBackend::decode_all_tokens`] once per pass.
///
/// All state mutation is local to the implementor — the session never
/// constructs / inspects KV caches, so per-model cache lifetime stays
/// inside per-model code.
///
/// `confirmed_token_ids` is the seed prefix the decoder should
/// re-replay before sampling new tokens (lets the cache warm up
/// without re-running the audio encoder). The returned `Vec<u32>` is
/// the **full** token sequence (confirmed prefix + newly sampled
/// tail). Implementors that don't need the replay-replay-then-sample
/// optimization can ignore `confirmed_token_ids` and return only the
/// newly sampled tokens with `confirmed_token_ids` prepended; the
/// session uses `confirmed_token_ids.len()` as the split point.
pub trait StreamingDecoderBackend {
  /// Run one decode pass over `audio_features`, returning the full
  /// token-id sequence (confirmed seed + newly sampled tokens).
  ///
  /// `max_tokens` is the caller's per-pass budget — implementations
  /// MUST stop sampling at this count even if EOS hasn't been
  /// reached, to bound per-pass latency.
  ///
  /// # Errors
  /// Implementation-defined — surfaced via [`Result`].
  fn decode_all_tokens(
    &self,
    audio_features: &Array,
    confirmed_token_ids: &[u32],
    config: &StreamingConfig,
    max_tokens: usize,
  ) -> Result<Vec<u32>>;
}

/// Architecture-specific tokenizer bridge for streaming detok.
///
/// The session only needs to convert id-slices to display text
/// incrementally — it never encodes. Per-model code typically wires
/// this through [`crate::tokenizer::sentencepiece::SentencePieceTokenizer`]
/// or the [`crate::tokenizer::Tokenizer`] HF wrapper.
pub trait StreamingTokenizer {
  /// Decode an id sequence to displayable text.
  fn decode_ids(&self, ids: &[u32]) -> String;
}

/// Streaming-decode pending state, mirroring Swift's
/// `SessionSharedState`. Owned by the session (no lock — single-thread
/// access).
#[derive(Debug, Default)]
struct SessionSharedState {
  /// Accumulated text from completed encoder windows — frozen, never
  /// re-decoded.
  completed_text: String,
  /// Confirmed-prefix tokens for the current pending window.
  confirmed_token_ids: Vec<u32>,
  /// Provisional tail under agreement-tracking.
  provisional_token_ids: Vec<u32>,
  /// First-seen `Instant` per provisional token — drives the
  /// `delay_ms` promotion clock.
  provisional_first_seen: Vec<Instant>,
  /// Per-provisional consecutive agreement counters.
  provisional_agreement_counts: Vec<usize>,
  /// Display string for the confirmed prefix.
  confirmed_text: String,
}

/// Per-decode-pass parameter bundle. Lets the helper functions stay
/// small and avoids cloning the session into every call.
struct DecodePassParams<'a> {
  audio_features: &'a Array,
  confirmed_token_ids: Vec<u32>,
  display_prefix: String,
  prev_provisional: Vec<u32>,
  prev_first_seen: Vec<Instant>,
  prev_agreement_counts: Vec<usize>,
  min_agreement_passes: usize,
}

/// Synchronous streaming-STT orchestration session.
///
/// Generic over the per-architecture encoder backend `B`, decoder
/// backend `D`, and tokenizer `T`. Owns its own
/// [`IncrementalMelSpectrogram`] + [`StreamingEncoder`].
pub struct StreamingInferenceSession<B, D, T>
where
  B: StreamingEncoderBackend,
  D: StreamingDecoderBackend,
  T: StreamingTokenizer,
{
  decoder: D,
  tokenizer: T,
  config: StreamingConfig,

  mel_processor: IncrementalMelSpectrogram,
  encoder: StreamingEncoder<B>,

  shared: SessionSharedState,
  is_active: bool,
  total_samples_fed: usize,
  last_decode_time: Option<Instant>,
  boundary_fast_decode_until: Option<Instant>,
  has_new_encoder_content: bool,
  /// Number of encoder windows whose text has been frozen into
  /// `completed_text`.
  frozen_window_count: usize,
  /// Retryable queue of newly-encoded full windows pending a finalize
  /// decode. A decoder error during finalization leaves the failed
  /// window at the front; the next `feed_audio` / `stop` call retries
  /// it from there. Drained one window at a time as decodes succeed
  /// (`frozen_window_count` advances in lock-step).
  pending_finalize_queue: VecDeque<Array>,
}

impl<B, D, T> StreamingInferenceSession<B, D, T>
where
  B: StreamingEncoderBackend,
  D: StreamingDecoderBackend,
  T: StreamingTokenizer,
{
  /// Build a new session. `sample_rate` and `n_mels` describe the
  /// mel-extractor configuration that the encoder backend expects;
  /// `overlap_frames` is the encoder window's cross-window overlap
  /// in mel frames (matches Swift's `overlapFrames`). Per the Swift
  /// reference, `n_fft = 400` and `hop_length = 160` are fixed for
  /// the streaming mel extractor.
  ///
  /// # Errors
  /// Propagates from [`IncrementalMelSpectrogram::new`].
  pub fn new(
    decoder: D,
    tokenizer: T,
    config: StreamingConfig,
    encoder_backend: B,
    sample_rate: u32,
    n_mels: usize,
    overlap_frames: usize,
  ) -> Result<Self> {
    let mel_processor = IncrementalMelSpectrogram::new(sample_rate, 400, 160, n_mels)?;
    let max_cached_windows = config.max_cached_windows;
    let encoder = StreamingEncoder::new(encoder_backend, max_cached_windows, overlap_frames);
    Ok(Self {
      decoder,
      tokenizer,
      config,
      mel_processor,
      encoder,
      shared: SessionSharedState::default(),
      is_active: true,
      total_samples_fed: 0,
      last_decode_time: None,
      boundary_fast_decode_until: None,
      has_new_encoder_content: false,
      frozen_window_count: 0,
      pending_finalize_queue: VecDeque::new(),
    })
  }

  /// Borrow the underlying [`StreamingConfig`].
  pub fn config(&self) -> &StreamingConfig {
    &self.config
  }

  /// Total samples fed since construction / last [`reset`](Self::reset).
  pub fn total_samples_fed(&self) -> usize {
    self.total_samples_fed
  }

  /// Number of fully encoded windows.
  pub fn encoded_window_count(&self) -> usize {
    self.encoder.encoded_window_count()
  }

  /// Whether the session is still active (not stopped / cancelled).
  pub fn is_active(&self) -> bool {
    self.is_active
  }

  /// Feed audio samples + run a decode pass when the cadence/boundary
  /// rules dictate. Returns the events emitted during this call —
  /// empty `Vec` when no decode runs.
  ///
  /// # Errors
  /// Propagates from the mel processor / encoder / decoder backend.
  pub fn feed_audio(&mut self, samples: &[f32]) -> Result<Vec<TranscriptionEvent>> {
    if !self.is_active {
      return Ok(Vec::new());
    }

    self.total_samples_fed = self.total_samples_fed.saturating_add(samples.len());

    let Some(mel_frames) = self.mel_processor.process(samples)? else {
      return Ok(Vec::new());
    };
    let new_windows = self.encoder.feed(&mel_frames)?;
    if new_windows > 0 || self.encoder.has_pending_frames() {
      self.has_new_encoder_content = true;
    }

    let now = Instant::now();
    if new_windows > 0 {
      let boost = self.config.boundary_boost_seconds.max(0.0);
      if boost > 0.0 {
        self.boundary_fast_decode_until = Some(now + std::time::Duration::from_secs_f64(boost));
      } else {
        self.boundary_fast_decode_until = None;
      }
    }

    let effective_decode_interval_seconds = if let Some(until) = self.boundary_fast_decode_until
      && now < until
    {
      let fast = self.config.boundary_decode_interval_seconds.max(0.05);
      let normal = self.config.decode_interval_seconds.max(0.05);
      fast.min(normal)
    } else {
      self.boundary_fast_decode_until = None;
      self.config.decode_interval_seconds.max(0.05)
    };

    // F3: a non-empty retry queue carries the implicit promise of
    // "we owe the caller a finalize-decode result on this audio."
    // Drain it on every feed (the inner loop is a noop when the queue
    // is empty).
    let has_pending_retries =
      self.config.finalize_completed_windows && !self.pending_finalize_queue.is_empty();

    let should_decode =
      if (self.config.finalize_completed_windows && new_windows > 0) || has_pending_retries {
        true
      } else if let Some(last) = self.last_decode_time {
        now.duration_since(last).as_secs_f64() >= effective_decode_interval_seconds
      } else {
        self.has_new_encoder_content
      };

    if should_decode && (self.has_new_encoder_content || has_pending_retries) {
      self.has_new_encoder_content = false;
      let is_boundary_finalize_pass = self.config.finalize_completed_windows && new_windows > 0;
      if !is_boundary_finalize_pass {
        self.last_decode_time = Some(now);
      }
      return self.run_decode_pass();
    }

    Ok(Vec::new())
  }

  /// Flush pending samples + run the final decode pass + emit the
  /// terminal [`TranscriptionEvent::Ended`] event.
  ///
  /// After `stop`, [`is_active`](Self::is_active) returns `false`. A
  /// follow-up `feed_audio` is a no-op.
  ///
  /// # Errors
  /// Propagates from the mel processor / encoder / decoder backend.
  pub fn stop(&mut self) -> Result<Vec<TranscriptionEvent>> {
    if !self.is_active {
      return Ok(Vec::new());
    }
    self.is_active = false;

    let mut events: Vec<TranscriptionEvent> = Vec::new();

    // Drain the mel processor for the final overlap-pad frames.
    if let Some(mel_frames) = self.mel_processor.flush()? {
      let _ = self.encoder.feed(&mel_frames)?;
    }

    // If finalize_completed_windows is on, decode any newly-encoded full
    // windows as one-shot finalize passes first. Drain into the retry
    // queue (F3) so a decoder error doesn't silently drop the window.
    if self.config.finalize_completed_windows {
      let drained = self.encoder.drain_newly_encoded_windows();
      for window in drained {
        self.pending_finalize_queue.push_back(window);
      }
      if !self.pending_finalize_queue.is_empty() {
        events.extend(self.finalize_completed_windows()?);
      }
    } else {
      self.freeze_completed_windows();
    }

    // Decode the pending partial window — if any.
    if let Some(audio_features) = self.encoder.encode_pending()? {
      if audio_features.shape().first().copied().unwrap_or(0) > 0 {
        let display_prefix = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
        let confirmed_count = self.shared.confirmed_token_ids.len();
        let estimated_tokens = self
          .config
          .max_tokens_per_pass
          .min(confirmed_count.saturating_add(24).max(24));
        let token_ids = self.decoder.decode_all_tokens(
          &audio_features,
          &self.shared.confirmed_token_ids,
          &self.config,
          estimated_tokens,
        )?;
        // Final text rolls everything into confirmed.
        self.shared.confirmed_token_ids = token_ids;
        self.shared.provisional_token_ids.clear();
        self.shared.provisional_first_seen.clear();
        self.shared.provisional_agreement_counts.clear();
        self.shared.confirmed_text = self.tokenizer.decode_ids(&self.shared.confirmed_token_ids);
        let _ = display_prefix; // computed for parity; not needed after final replace
      }
    } else {
      // No pending frames — promote provisional to confirmed.
      if !self.shared.provisional_token_ids.is_empty() {
        let promoted = std::mem::take(&mut self.shared.provisional_token_ids);
        self.shared.confirmed_token_ids.extend(promoted);
        self.shared.provisional_first_seen.clear();
        self.shared.provisional_agreement_counts.clear();
      }
      if !self.shared.confirmed_token_ids.is_empty() {
        self.shared.confirmed_text = self.tokenizer.decode_ids(&self.shared.confirmed_token_ids);
      }
    }

    let final_text = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
    events.push(TranscriptionEvent::Ended {
      full_text: final_text,
    });

    // Reset state per Swift's `finishStop` tail.
    self.encoder.reset();
    self.mel_processor.reset();
    self.boundary_fast_decode_until = None;

    Ok(events)
  }

  /// Cancel without producing the final `.ended` event — used for
  /// abandoned sessions.
  pub fn cancel(&mut self) {
    self.is_active = false;
    self.encoder.reset();
    self.mel_processor.reset();
    self.boundary_fast_decode_until = None;
    self.shared = SessionSharedState::default();
    self.pending_finalize_queue.clear();
  }

  /// Reset all state for a fresh session.
  pub fn reset(&mut self) {
    self.is_active = true;
    self.total_samples_fed = 0;
    self.last_decode_time = None;
    self.boundary_fast_decode_until = None;
    self.has_new_encoder_content = false;
    self.frozen_window_count = 0;
    self.encoder.reset();
    self.mel_processor.reset();
    self.shared = SessionSharedState::default();
    self.pending_finalize_queue.clear();
  }

  // -------------------------------------------------------------------
  // Internal: decode-pass orchestration
  // -------------------------------------------------------------------

  fn run_decode_pass(&mut self) -> Result<Vec<TranscriptionEvent>> {
    // If finalize_completed_windows is on AND we have newly-encoded
    // full windows, do a one-shot finalize on each, then continue.
    //
    // F3: never drain newly-encoded windows out of the system in a path
    // that can't replay them. Push freshly-drained windows into the
    // retry queue; `finalize_completed_windows` then pops them one at a
    // time as decodes succeed (and leaves any failed window at the
    // front for the next pass to retry).
    if self.config.finalize_completed_windows {
      let drained = self.encoder.drain_newly_encoded_windows();
      for window in drained {
        self.pending_finalize_queue.push_back(window);
      }
      if !self.pending_finalize_queue.is_empty() {
        return self.finalize_completed_windows();
      }
    } else {
      self.freeze_completed_windows();
    }

    // Only decode the current pending (partial) window.
    let Some(audio_features) = self.encoder.encode_pending()? else {
      return Ok(Vec::new());
    };
    let num_audio_tokens = audio_features.shape().first().copied().unwrap_or(0);
    if num_audio_tokens == 0 {
      return Ok(Vec::new());
    }

    let confirmed_count = self.shared.confirmed_token_ids.len();
    let windowed_seconds = num_audio_tokens as f64 / 13.0;
    let estimated_total_tokens = ((windowed_seconds * 10.0).ceil() as usize).max(24);
    let max_tokens = self
      .config
      .max_tokens_per_pass
      .min(estimated_total_tokens.max(confirmed_count.saturating_add(24)));

    let display_prefix = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
    let min_agreement_passes = if let Some(until) = self.boundary_fast_decode_until
      && Instant::now() < until
    {
      self
        .config
        .min_agreement_passes
        .max(self.config.boundary_min_agreement_passes)
        .max(1)
    } else {
      self.config.min_agreement_passes.max(1)
    };

    let params = DecodePassParams {
      audio_features: &audio_features,
      confirmed_token_ids: self.shared.confirmed_token_ids.clone(),
      display_prefix,
      prev_provisional: self.shared.provisional_token_ids.clone(),
      prev_first_seen: self.shared.provisional_first_seen.clone(),
      prev_agreement_counts: self.shared.provisional_agreement_counts.clone(),
      min_agreement_passes,
    };

    let start = Instant::now();
    let all_token_ids = self.decoder.decode_all_tokens(
      params.audio_features,
      &params.confirmed_token_ids,
      &self.config,
      max_tokens,
    )?;
    let decode_time = start.elapsed().as_secs_f64();

    Ok(self.promote_tokens(&all_token_ids, &params, decode_time))
  }

  fn promote_tokens(
    &mut self,
    all_token_ids: &[u32],
    params: &DecodePassParams<'_>,
    decode_time: f64,
  ) -> Vec<TranscriptionEvent> {
    let confirmed_count = params.confirmed_token_ids.len();
    let new_provisional: Vec<u32> = all_token_ids
      .iter()
      .skip(confirmed_count)
      .copied()
      .collect();
    let gen_token_count = all_token_ids.len();
    let now = Instant::now();
    let delay = std::time::Duration::from_millis(u64::from(self.config.delay_preset.delay_ms()));

    // Common prefix match-length between prev provisional and new.
    let mut match_len = 0;
    let compare_len = params.prev_provisional.len().min(new_provisional.len());
    for (i, new_id) in new_provisional.iter().enumerate().take(compare_len) {
      if params.prev_provisional[i] == *new_id {
        match_len = i + 1;
      } else {
        break;
      }
    }

    let mut next_first_seen: Vec<Instant> = Vec::with_capacity(new_provisional.len());
    let mut next_agreement_counts: Vec<usize> = Vec::with_capacity(new_provisional.len());
    for i in 0..new_provisional.len() {
      if i < match_len {
        let seen = params.prev_first_seen.get(i).copied().unwrap_or(now);
        let prev_agreement = params.prev_agreement_counts.get(i).copied().unwrap_or(1);
        next_first_seen.push(seen);
        next_agreement_counts.push(prev_agreement.saturating_add(1).max(1));
      } else {
        next_first_seen.push(now);
        next_agreement_counts.push(1);
      }
    }

    let required_agreement_passes = params.min_agreement_passes.max(1);
    let mut promotion_count = 0;
    for i in 0..new_provisional.len() {
      let has_delay = next_first_seen
        .get(i)
        .map(|t| now.duration_since(*t) >= delay)
        .unwrap_or(false);
      let has_agreement = next_agreement_counts
        .get(i)
        .map(|c| *c >= required_agreement_passes)
        .unwrap_or(false);
      if has_delay && has_agreement {
        promotion_count = i + 1;
      } else {
        break;
      }
    }

    let final_provisional: Vec<u32> = new_provisional
      .iter()
      .skip(promotion_count)
      .copied()
      .collect();
    let final_first_seen: Vec<Instant> = next_first_seen
      .iter()
      .skip(promotion_count)
      .copied()
      .collect();
    let final_agreement_counts: Vec<usize> = next_agreement_counts
      .iter()
      .skip(promotion_count)
      .copied()
      .collect();

    let mut events: Vec<TranscriptionEvent> = Vec::new();
    if promotion_count > 0 {
      let promoted: Vec<u32> = new_provisional[..promotion_count].to_vec();
      self.shared.confirmed_token_ids.extend(promoted);
      self.shared.confirmed_text = self.tokenizer.decode_ids(&self.shared.confirmed_token_ids);
      events.push(TranscriptionEvent::Confirmed {
        text: concat_text(&self.shared.completed_text, &self.shared.confirmed_text),
      });
    }
    self.shared.provisional_token_ids = final_provisional.clone();
    self.shared.provisional_first_seen = final_first_seen;
    self.shared.provisional_agreement_counts = final_agreement_counts;

    let final_prov_text = self.tokenizer.decode_ids(&final_provisional);
    let display_prefix = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
    events.push(TranscriptionEvent::DisplayUpdate {
      confirmed_text: display_prefix,
      provisional_text: final_prov_text,
    });
    let _ = params.display_prefix; // shape parity — used only for the streaming preview event

    let total_audio_seconds = self.total_samples_fed as f64 / 16_000.0;
    let tps = if decode_time > 0.0 {
      gen_token_count as f64 / decode_time
    } else {
      0.0
    };
    events.push(TranscriptionEvent::Stats(StreamingStats {
      encoded_window_count: self.encoder.encoded_window_count(),
      total_audio_seconds,
      tokens_per_second: tps,
      real_time_factor: 0.0,
      peak_memory_gb: peak_memory_gb_or_zero(),
    }));
    events
  }

  /// Finalize the windows in `pending_finalize_queue`: run a fresh
  /// decode over each, append its text to `completed_text`, and reset
  /// the streaming decode state.
  ///
  /// F2: ALWAYS run `decoder.decode_all_tokens` for finalized windows.
  /// The previously-streamed provisional/confirmed text is consulted
  /// only as an explicit fallback when the full decode for the first
  /// queued window returns empty text — otherwise the streamed
  /// preview's partial-text would freeze in place and the rest of the
  /// boundary audio would be dropped.
  ///
  /// F3: pops one window at a time, advancing `frozen_window_count`
  /// after each successful append. On `Err` the failed window is left
  /// at the queue front so a subsequent `feed_audio` / `stop` call can
  /// retry it without losing already-encoded audio.
  fn finalize_completed_windows(&mut self) -> Result<Vec<TranscriptionEvent>> {
    if self.pending_finalize_queue.is_empty() {
      return Ok(Vec::new());
    }
    let mut total_decode_time: f64 = 0.0;
    let mut total_generated_tokens: usize = 0;
    // F2: capture the streamed-text fallback for the first window only,
    // gated on actually being the first call after a non-empty shared
    // state. Consumed via `take()` after the first pop so retries don't
    // re-apply it.
    let mut streamed_fallback_for_first_window: Option<String> = {
      let mut stream_tokens: Vec<u32> = self.shared.confirmed_token_ids.clone();
      stream_tokens.extend(self.shared.provisional_token_ids.iter().copied());
      if stream_tokens.is_empty() {
        None
      } else {
        Some(self.tokenizer.decode_ids(&stream_tokens))
      }
    };

    let mut events: Vec<TranscriptionEvent> = Vec::new();
    while let Some(audio_features) = self.pending_finalize_queue.front() {
      let num_audio_tokens = audio_features.shape().first().copied().unwrap_or(0);
      let candidate_fallback = streamed_fallback_for_first_window.take();
      let selected_window_text = if num_audio_tokens == 0 {
        // Empty audio: skip decode but allow the streamed fallback to
        // carry text forward (rare — guards against zero-row boundary
        // windows the encoder occasionally produces).
        candidate_fallback.unwrap_or_default()
      } else {
        let start = Instant::now();
        // F2 + F3: ALWAYS attempt the full decode. On `Err` the `?`
        // propagates up; the queue front is unchanged so the next
        // pass retries this window. `frozen_window_count` has NOT
        // advanced yet, preserving the invariant
        // `frozen_window_count == encoded_window_count - queue.len()`.
        let token_ids = self.decoder.decode_all_tokens(
          audio_features,
          &[],
          &self.config,
          self.config.max_tokens_per_pass,
        )?;
        let decode_time = start.elapsed().as_secs_f64();
        total_decode_time += decode_time;
        total_generated_tokens = total_generated_tokens.saturating_add(token_ids.len());
        let full_text = self.tokenizer.decode_ids(&token_ids);
        // F2: only fall back to streamed text when the FULL decode
        // produced nothing. Otherwise the full decode wins — including
        // for the first window — so we never freeze stale partial text
        // over freshly-encoded boundary audio.
        if full_text.trim().is_empty()
          && let Some(fallback) = candidate_fallback
        {
          fallback
        } else {
          full_text
        }
      };
      // Decode succeeded (or there was no audio to decode): commit
      // this window now, clear shared streaming state, advance the
      // frozen-window counter, and pop the queue.
      if !selected_window_text.trim().is_empty() {
        append_text(&selected_window_text, &mut self.shared.completed_text);
      }
      self.shared.confirmed_token_ids.clear();
      self.shared.provisional_token_ids.clear();
      self.shared.provisional_first_seen.clear();
      self.shared.provisional_agreement_counts.clear();
      self.shared.confirmed_text.clear();
      self.pending_finalize_queue.pop_front();
      self.frozen_window_count = self.frozen_window_count.saturating_add(1);
    }

    let total_audio_seconds = self.total_samples_fed as f64 / 16_000.0;
    let tps = if total_decode_time > 0.0 {
      total_generated_tokens as f64 / total_decode_time
    } else {
      0.0
    };
    events.push(TranscriptionEvent::Stats(StreamingStats {
      encoded_window_count: self.encoder.encoded_window_count(),
      total_audio_seconds,
      tokens_per_second: tps,
      real_time_factor: 0.0,
      peak_memory_gb: peak_memory_gb_or_zero(),
    }));
    Ok(events)
  }

  fn freeze_completed_windows(&mut self) {
    let current = self.encoder.encoded_window_count();
    if current <= self.frozen_window_count {
      return;
    }
    let mut all_tokens: Vec<u32> = self.shared.confirmed_token_ids.clone();
    all_tokens.extend(self.shared.provisional_token_ids.iter().copied());
    if !all_tokens.is_empty() {
      let window_text = self.tokenizer.decode_ids(&all_tokens);
      append_text(&window_text, &mut self.shared.completed_text);
    }
    self.shared.confirmed_token_ids.clear();
    self.shared.provisional_token_ids.clear();
    self.shared.provisional_first_seen.clear();
    self.shared.provisional_agreement_counts.clear();
    self.shared.confirmed_text.clear();
    self.frozen_window_count = current;
  }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Append `segment` to `base` with whitespace handling — mirrors
/// Swift's `appendText`'s `trimmingCharacters(in: .whitespacesAndNewlines)`
/// plus the leading-space insertion when both halves are non-empty and
/// neither side already supplies the boundary whitespace. Simplified
/// (no deduping) — the Swift reference's dedupe heuristics are
/// decode-quality polish, not orchestration semantics. Reuse via
/// [`concat_text`].
fn append_text(segment: &str, base: &mut String) {
  let trimmed = segment.trim();
  if trimmed.is_empty() {
    return;
  }
  if base.is_empty() {
    base.push_str(trimmed);
    return;
  }
  let base_last_is_ws = base.chars().last().is_some_and(char::is_whitespace);
  let seg_first_is_ws = trimmed.chars().next().is_some_and(char::is_whitespace);
  if base_last_is_ws || seg_first_is_ws {
    base.push_str(trimmed);
  } else {
    base.push(' ');
    base.push_str(trimmed);
  }
}

fn concat_text(a: &str, b: &str) -> String {
  let mut out = String::with_capacity(a.len() + b.len() + 1);
  out.push_str(a);
  append_text(b, &mut out);
  out
}

/// Wrapper around [`crate::memory::peak_memory`] that returns
/// `peak / 1e9` GB or `0.0` if the read errors. Mirrors the Swift
/// reference's `Double(Memory.peakMemory) / 1e9` formula.
fn peak_memory_gb_or_zero() -> f64 {
  crate::memory::peak_memory()
    .map(|bytes| bytes as f64 / 1e9)
    .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::audio::stt::streaming::{encoder::StreamingEncoderBackend, types::DelayPreset};
  use std::sync::Mutex;

  // -----------------------------------------------------------------
  // Mocks
  // -----------------------------------------------------------------

  struct MockEncoder {
    window_size: usize,
  }

  impl StreamingEncoderBackend for MockEncoder {
    fn window_size(&self) -> usize {
      self.window_size
    }

    fn encode_window(&self, mel_window: &Array, _valid_frames: usize) -> Result<Array> {
      let rows = mel_window.shape().first().copied().unwrap_or(0);
      let buf = vec![0.0_f32; rows * 2];
      Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
    }
  }

  struct MockDecoder {
    /// Per-call returned token sequence.
    tokens: Mutex<Vec<Vec<u32>>>,
    /// Records `(rows, confirmed_count, max_tokens)` per call.
    calls: Mutex<Vec<(usize, usize, usize)>>,
  }

  impl MockDecoder {
    fn with_tokens(seqs: Vec<Vec<u32>>) -> Self {
      Self {
        tokens: Mutex::new(seqs),
        calls: Mutex::new(Vec::new()),
      }
    }

    fn call_count(&self) -> usize {
      self.calls.lock().unwrap().len()
    }
  }

  impl StreamingDecoderBackend for MockDecoder {
    fn decode_all_tokens(
      &self,
      audio_features: &Array,
      confirmed_token_ids: &[u32],
      _config: &StreamingConfig,
      max_tokens: usize,
    ) -> Result<Vec<u32>> {
      let rows = audio_features.shape().first().copied().unwrap_or(0);
      self
        .calls
        .lock()
        .unwrap()
        .push((rows, confirmed_token_ids.len(), max_tokens));
      let mut queue = self.tokens.lock().unwrap();
      let next = if queue.is_empty() {
        Vec::new()
      } else {
        queue.remove(0)
      };
      Ok(next)
    }
  }

  struct MockTokenizer;
  impl StreamingTokenizer for MockTokenizer {
    fn decode_ids(&self, ids: &[u32]) -> String {
      ids
        .iter()
        .map(|id| format!("t{id}"))
        .collect::<Vec<_>>()
        .join(" ")
    }
  }

  fn audio_session() -> StreamingInferenceSession<MockEncoder, MockDecoder, MockTokenizer> {
    let cfg = StreamingConfig {
      // Force decode on every feed for deterministic test timing.
      decode_interval_seconds: 0.0,
      boundary_decode_interval_seconds: 0.0,
      boundary_boost_seconds: 0.0,
      max_cached_windows: 4,
      finalize_completed_windows: false,
      min_agreement_passes: 1,
      boundary_min_agreement_passes: 1,
      delay_preset: DelayPreset::Custom(0),
      ..StreamingConfig::default()
    };
    StreamingInferenceSession::new(
      MockDecoder::with_tokens(vec![vec![10, 11, 12]]),
      MockTokenizer,
      cfg,
      MockEncoder { window_size: 16 },
      16_000,
      8,
      0,
    )
    .unwrap()
  }

  #[test]
  fn feed_audio_short_input_yields_no_events_until_mel_emits() {
    let mut session = audio_session();
    // A single short feed should not invoke the encoder/decoder if mel
    // has produced no frames yet.
    let events = session.feed_audio(&[0.0_f32; 1]).unwrap();
    // No decode pass with this tiny input.
    assert!(events.is_empty(), "events={events:?}");
  }

  #[test]
  fn feed_audio_long_input_drives_pending_window_decode_with_recorded_call_shape() {
    let mut session = audio_session();
    // Feed enough samples that mel emits a few frames (but not a full
    // window of 16). Hop=160, n_fft=400 → ~2000 samples gives ~10
    // mel frames.
    let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
    let events = session.feed_audio(&samples).unwrap();
    // Decode should have been called once with rows = pending frame
    // count (encoder mock returns rows passthrough).
    assert_eq!(session.decoder.call_count(), 1);
    let calls = session.decoder.calls.lock().unwrap();
    let (rows, confirmed_count, _max_tokens) = calls[0];
    assert!(rows > 0, "expected non-zero pending rows, got {rows}");
    assert_eq!(confirmed_count, 0);
    drop(calls);
    // Promote-immediate config (delay 0, agreement 1) → tokens
    // confirmed in one pass → events should contain Confirmed +
    // DisplayUpdate + Stats.
    assert!(
      matches!(events.first(), Some(TranscriptionEvent::Confirmed { .. })),
      "events[0]={:?}",
      events.first()
    );
    assert!(
      events
        .iter()
        .any(|e| matches!(e, TranscriptionEvent::Stats(_)))
    );
  }

  #[test]
  fn stop_emits_ended_event_with_accumulated_text() {
    let mut session = audio_session();
    let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();
    let stop_events = session.stop().unwrap();
    // Last event must be Ended.
    assert!(
      matches!(stop_events.last(), Some(TranscriptionEvent::Ended { .. })),
      "stop events: {stop_events:?}"
    );
    assert!(!session.is_active());
  }

  #[test]
  fn cancel_marks_inactive_and_drops_state() {
    let mut session = audio_session();
    let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();
    session.cancel();
    assert!(!session.is_active());
    // Follow-up feed_audio is a no-op.
    let after = session.feed_audio(&samples).unwrap();
    assert!(after.is_empty());
  }

  #[test]
  fn append_text_basic_concatenation_and_trim() {
    let mut base = String::new();
    append_text("hello", &mut base);
    assert_eq!(base, "hello");
    append_text("world", &mut base);
    assert_eq!(base, "hello world");
    append_text("  ", &mut base);
    assert_eq!(base, "hello world");
    append_text("!", &mut base);
    assert_eq!(base, "hello world !");
  }

  // ----------------------------------------------------------------
  // F2 / F3 — completed-window finalization regressions
  // ----------------------------------------------------------------

  /// Mock decoder that takes a `Result<Vec<u32>>` per call so individual
  /// passes can inject decoder errors. Tracks call count for retry
  /// verification.
  struct ScriptedDecoder {
    /// Per-call scripted results (front-popped). Empty after exhaustion
    /// → defaults to `Ok(vec![])`.
    results: Mutex<Vec<Result<Vec<u32>>>>,
    /// Count of `decode_all_tokens` invocations.
    calls: Mutex<usize>,
  }

  impl ScriptedDecoder {
    fn with_results(results: Vec<Result<Vec<u32>>>) -> Self {
      Self {
        results: Mutex::new(results),
        calls: Mutex::new(0),
      }
    }

    fn call_count(&self) -> usize {
      *self.calls.lock().unwrap()
    }
  }

  impl StreamingDecoderBackend for ScriptedDecoder {
    fn decode_all_tokens(
      &self,
      _audio_features: &Array,
      _confirmed_token_ids: &[u32],
      _config: &StreamingConfig,
      _max_tokens: usize,
    ) -> Result<Vec<u32>> {
      *self.calls.lock().unwrap() += 1;
      let mut q = self.results.lock().unwrap();
      if q.is_empty() {
        Ok(Vec::new())
      } else {
        q.remove(0)
      }
    }
  }

  /// Build a streaming-finalize-enabled session over a scripted decoder.
  /// Window size 8 mel-frames keeps the boundary cadence tight so the
  /// tests don't need wall-clock waits.
  fn finalize_session(
    decoder: ScriptedDecoder,
  ) -> StreamingInferenceSession<MockEncoder, ScriptedDecoder, MockTokenizer> {
    let cfg = StreamingConfig {
      decode_interval_seconds: 0.0,
      boundary_decode_interval_seconds: 0.0,
      boundary_boost_seconds: 0.0,
      max_cached_windows: 8,
      finalize_completed_windows: true,
      min_agreement_passes: 1,
      boundary_min_agreement_passes: 1,
      delay_preset: DelayPreset::Custom(0),
      ..StreamingConfig::default()
    };
    StreamingInferenceSession::new(
      decoder,
      MockTokenizer,
      cfg,
      MockEncoder { window_size: 8 },
      16_000,
      8,
      0,
    )
    .unwrap()
  }

  /// Drive the session through (a) a partial-window feed that exercises
  /// the pending-window pre-boundary decode, then (b) a top-up feed
  /// that closes the first window and triggers the boundary finalize
  /// pass. Window-size 8 mel-frames; n_fft=400, hop=160 → 400 + 7×160
  /// = 1 520 samples for one window of mel content. Using 800 + 1 200
  /// samples gives one partial-decode call + one finalize-decode call.
  ///
  /// Returns `(partial_events_result, boundary_events_result)` so the
  /// F3 tests can deliberately drive the decoder past a scripted `Err`
  /// without panicking on `.unwrap()` and observe each phase's events
  /// (or error) independently.
  fn drive_two_phase(
    session: &mut StreamingInferenceSession<MockEncoder, ScriptedDecoder, MockTokenizer>,
  ) -> (
    Result<Vec<TranscriptionEvent>>,
    Result<Vec<TranscriptionEvent>>,
  ) {
    let partial: Vec<f32> = (0..800).map(|i| (i as f32 * 0.001).sin()).collect();
    let topup: Vec<f32> = (800..2_000).map(|i| (i as f32 * 0.001).sin()).collect();
    let partial_events = session.feed_audio(&partial);
    let boundary_events = session.feed_audio(&topup);
    (partial_events, boundary_events)
  }

  /// F2: when the first window is finalized, the FULL decode must run
  /// over the completed-window features — even when streamed-fallback
  /// text exists from a prior pending-window decode. The streamed
  /// fallback only applies as an EMPTY-decode tiebreaker.
  #[test]
  fn streaming_session_first_window_finalization_runs_full_decode_not_streamed_fallback() {
    // First call (partial-window decode while encoder is filling):
    //   returns one token id → seeds provisional/confirmed state.
    // Second call (the full-window finalize): returns a different,
    //   non-empty result. F2 requires that this call HAPPEN.
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1, 2, 3]), Ok(vec![9, 8, 7])]);
    let mut session = finalize_session(decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    boundary_events.unwrap();
    // 2 decode calls: one for the pre-boundary pending-window decode
    // (when the first partial buffer arrived) + one for the boundary
    // finalize-pass on the completed window. The boundary finalize MUST
    // execute, even with provisional/confirmed state present.
    assert!(
      session.decoder.call_count() >= 2,
      "expected >= 2 decoder calls (pending + finalize), got {}",
      session.decoder.call_count()
    );
  }

  /// F2: when the streamed-fallback text differs from the full-decode
  /// text, the FULL-decode text MUST land in `completed_text` — the
  /// stale partial preview is never frozen over fresh boundary audio.
  #[test]
  fn streaming_session_first_window_finalization_appends_full_decode_not_partial_text() {
    // Pending-decode returns one set of tokens (tokens 1..3 → "t1 t2 t3")
    // → first pass promotes them into `confirmed_text`.
    // Boundary finalize returns a different set (tokens 90, 91, 92 →
    // "t90 t91 t92"). The Ended `full_text` must contain the
    // full-decode tokens, NOT the pending-text snapshot.
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1, 2, 3]), Ok(vec![90, 91, 92])]);
    let mut session = finalize_session(decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    boundary_events.unwrap();
    let stop_events = session.stop().unwrap();
    let TranscriptionEvent::Ended { full_text } = stop_events
      .last()
      .expect("expected Ended event at stop()")
      .clone()
    else {
      panic!("last stop event was not Ended: {stop_events:?}");
    };
    // The full-decode tokens (90 91 92) must dominate; the partial
    // preview "t1 t2 t3" must NOT be the entire text.
    assert!(
      full_text.contains("t90"),
      "Ended.full_text must include the full-decode text, got {full_text:?}"
    );
  }

  /// F3: on a finalize-decode error the failed window stays in the
  /// retry queue so a subsequent `feed_audio` call can re-attempt the
  /// decode (no encoder re-run, no silent audio loss).
  #[test]
  fn streaming_session_decoder_error_keeps_window_for_retry() {
    use crate::error::Error;
    let decoder = ScriptedDecoder::with_results(vec![
      // 1st: partial-window pending decode succeeds.
      Ok(vec![1]),
      // 2nd: boundary finalize fails.
      Err(Error::Backend {
        message: "scripted finalize failure".into(),
      }),
      // 3rd: retry succeeds.
      Ok(vec![42]),
    ]);
    let mut session = finalize_session(decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    // The scripted error must bubble up from the boundary feed.
    assert!(
      boundary_events.is_err(),
      "expected the scripted finalize Err to propagate, got {boundary_events:?}"
    );
    // The boundary finalize errored; queue still has the window.
    assert_eq!(
      session.pending_finalize_queue.len(),
      1,
      "errored finalize must leave the window in the retry queue"
    );
    // Retry: feed enough samples that we drive another decode cycle.
    // Even a small follow-up feed re-enters run_decode_pass via the
    // `has_pending_retries` short-circuit.
    let retry_events = session.feed_audio(&[0.0_f32; 200]).unwrap();
    // After retry the queue must be drained.
    assert_eq!(
      session.pending_finalize_queue.len(),
      0,
      "successful retry must pop the previously-failed window"
    );
    assert!(
      !retry_events.is_empty(),
      "retry decode must emit at least the Stats event"
    );
  }

  /// F3: while a finalize decode is errored / un-retried, the session's
  /// `frozen_window_count` MUST NOT advance — the invariant
  /// `frozen_window_count == encoded_window_count - queue.len()`
  /// holds across error/retry boundaries.
  #[test]
  fn streaming_session_decoder_error_does_not_advance_frozen_window_count() {
    use crate::error::Error;
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]), // partial pending decode
      Err(Error::Backend {
        message: "scripted finalize failure".into(),
      }),
    ]);
    let mut session = finalize_session(decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    assert!(
      boundary_events.is_err(),
      "expected the scripted finalize Err to propagate, got {boundary_events:?}"
    );
    // One encoder window completed; finalize errored → frozen MUST
    // still be 0.
    assert_eq!(
      session.encoded_window_count(),
      1,
      "encoder should have produced exactly one window"
    );
    assert_eq!(
      session.frozen_window_count, 0,
      "errored finalize must NOT advance frozen_window_count"
    );
    assert_eq!(session.pending_finalize_queue.len(), 1);
  }
}
