//! Unified retry-state machine for [`super::session::StreamingInferenceSession`].
//!
//! The streaming session orchestrates four fallible stages per call:
//! `mel.flush` → `encoder.feed` → per-window decode (`finalize` or
//! pending-window decode pass). Each stage may `Err` independently, and
//! the caller is allowed to retry the failed work by calling
//! `feed_audio` or `stop` again. Pre-`SessionRetryState`, the retry
//! plumbing for these stages lived in three independent session fields
//! (`pending_finalize_queue`, `pending_stop_mel_frames`,
//! `pending_bridge_drain_decode`) plus per-call locals; each new bypass
//! corner Codex review found required a fresh field/flag. Five
//! consecutive review rounds (R3 → R7) each found a NEW way for one of
//! those fields to desync from the others.
//!
//! [`SessionRetryState`] replaces the field-soup with a single source
//! of truth for in-flight retry obligations. Each fallible stage either
//! fully commits or sets a [`RetryStage`] that names exactly where the
//! next call must resume. The session's `discharge_retry_obligation`
//! method calls into the per-stage discharge helpers
//! ([`SessionRetryState::discharge_stop_encoder_feed`] etc.) at the
//! top of every `feed_audio` / `stop`, transactionally drives the
//! resume point, and only proceeds with new audio after the discharge
//! advances `resume_at` to `None`.

use std::collections::VecDeque;

use super::{
  encoder::{StreamingEncoder, StreamingEncoderBackend},
  mel_spectrogram::IncrementalMelSpectrogram,
};
use crate::{
  Array,
  error::{Error, Result},
};

/// One window of encoded mel that owes a finalize decode.
///
/// The `fallback_consumed` flag is a per-entry sticky bit set BEFORE
/// the fallible `decode_all_tokens` call so that on a decode `Err`,
/// the next retry sees `fallback_consumed == true` and gets no fallback
/// — stale streamed text from `SessionSharedState` is never re-applied.
/// Without this gate, a `decode_all_tokens` error would leave the
/// streamed text in `SessionSharedState`, and the retry's empty-decode
/// tiebreaker would freeze that stale provisional over fresh boundary
/// audio.
#[derive(Debug)]
pub(super) struct PendingFinalize {
  /// Encoded hidden states for the completed window.
  pub(super) encoder_output: Array,
  /// `true` once the streamed-text fallback has been offered for this
  /// entry. Sticky across retries — see the doc comment above.
  pub(super) fallback_consumed: bool,
}

/// Stage where a partial-failure retry should resume.
///
/// The streaming session's `feed_audio` / `stop` pipeline has multiple
/// distinct fallible stages. Pre-rewrite, a partial failure at any of
/// them required composing across multiple session fields to recover.
/// The unified state machine names each resume point explicitly so the
/// next call can dispatch to exactly the work that errored — no field
/// composition, no per-call locals that get lost on `?` propagation.
///
/// Failed finalize-queue decodes are NOT carried in a `RetryStage`
/// variant — the [`SessionRetryState::finalize_queue`] field's
/// non-emptiness is the obligation signal (the failed entry is at the
/// queue front).
#[derive(Debug)]
pub(super) enum RetryStage {
  /// `stop()`'s `mel.flush()` errored. The mel processor's transactional
  /// `flush` left its overlap buffer intact, so the next `stop()` call
  /// retries `mel.flush()` exactly. Carries no payload (the source-of-
  /// truth is `IncrementalMelSpectrogram::overlap_buffer`).
  StopMelFlush,
  /// `stop()`'s `mel.flush()` succeeded (committing-and-clearing the
  /// overlap buffer), and the freshly-flushed mel rows live nowhere but
  /// in `mel_frames` here. Any retry from `feed_audio` / `stop` MUST
  /// re-feed THIS array (the overlap is gone). On Ok the array is
  /// consumed and `resume_at` advances to the next stage if any.
  StopEncoderFeed { mel_frames: Array },
  /// One or more full encoder windows are committed to the encoder's
  /// `newly_encoded_windows` / `cached_windows` AND owe a decode pass.
  /// This covers two surfaces structurally:
  ///   (a) A previous call drained the [`StopEncoderFeed`] bridge with
  ///       a non-zero window count, then errored on a later step in the
  ///       same call (R6 corner: the count was a local, lost on `?`).
  ///   (b) A `run_decode_pass` invocation itself errored mid-way — the
  ///       windows are still in the encoder, the next call MUST decode
  ///       them BEFORE accepting new audio.
  /// Distinct from `resume_at = None + finalize_queue.is_empty()` —
  /// that state means no decode is owed.
  DecodeOwed,
  /// `stop()`'s post-finalize partial-window decode + Ended event
  /// emission. After this stage succeeds, `is_active` flips to false
  /// and the resume point clears. Carries the audio_features payload
  /// so the retry doesn't have to recompute encode_pending (which is
  /// itself fallible and idempotent — but skipping the recompute also
  /// avoids a redundant encoder forward pass).
  StopPartialDecode { audio_features: Option<Array> },
}

/// Unified retry-state machine.
///
/// Owns the finalize queue + the resume point. Discharge methods are
/// called at the top of every `feed_audio` / `stop` to drive any
/// pending obligation BEFORE new audio is touched; partial discharge
/// leaves [`has_obligation`](Self::has_obligation) true and the session
/// returns the events from what completed without accepting new work.
#[derive(Debug)]
pub(super) struct SessionRetryState {
  /// Resume point for the next call. `None` means no retry is owed —
  /// the session is in clean state. `Some(stage)` means the next
  /// `feed_audio` / `stop` MUST dispatch to that stage BEFORE
  /// processing new audio.
  resume_at: Option<RetryStage>,
  /// Per-window finalize-decode work, FIFO. Drained one window at a
  /// time as decodes succeed. A `decode_all_tokens` Err leaves the
  /// failed entry at the queue front; the queue's non-emptiness alone
  /// is the obligation signal (a non-empty queue ⇒
  /// [`has_obligation`](Self::has_obligation) returns `true` regardless
  /// of `resume_at`).
  finalize_queue: VecDeque<PendingFinalize>,
}

impl Default for SessionRetryState {
  fn default() -> Self {
    Self::new()
  }
}

impl SessionRetryState {
  /// Build a clean retry state — no obligation, empty queue.
  pub(super) fn new() -> Self {
    Self {
      resume_at: None,
      finalize_queue: VecDeque::new(),
    }
  }

  /// True iff some prior call left work that MUST be discharged before
  /// any new audio can be accepted. Either a `resume_at` is set OR the
  /// finalize queue is non-empty — both arms are equally blocking.
  pub(super) fn has_obligation(&self) -> bool {
    self.resume_at.is_some() || !self.finalize_queue.is_empty()
  }

  /// Inspect the resume point. Borrowed read-only — discharge methods
  /// mutate it via the dedicated `take_*` / `set_*` helpers below.
  pub(super) fn resume_at(&self) -> Option<&RetryStage> {
    self.resume_at.as_ref()
  }

  /// True iff `resume_at == Some(StopMelFlush)`. The session uses this
  /// to dispatch the unified `StopMelFlush` discharge — without it the
  /// `StopMelFlush` obligation would be stranded forever because
  /// `discharge_retry_obligation`'s dispatcher would have nothing to
  /// gate on and `has_obligation()` would short-circuit `stop()` to an
  /// early-return.
  pub(super) fn has_pending_stop_mel_flush(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::StopMelFlush))
  }

  /// True iff `resume_at` names a stage whose source-of-truth lives
  /// inside `mel_processor` / `encoder` — i.e. some prior call's
  /// encoder.feed errored and the staged mel rows live in
  /// `RetryStage::StopEncoderFeed { mel_frames }`. The session uses
  /// this to keep the contract "drain the staged stop-tail BEFORE
  /// processing new feed audio."
  pub(super) fn has_pending_stop_encoder_feed(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::StopEncoderFeed { .. }))
  }

  /// True iff `resume_at == Some(DecodeOwed)`.
  pub(super) fn has_decode_owed(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::DecodeOwed))
  }

  /// Borrow the finalize queue — the session needs read-only access to
  /// drive the `has_pending_retries` gate.
  pub(super) fn finalize_queue(&self) -> &VecDeque<PendingFinalize> {
    &self.finalize_queue
  }

  /// Mutable access to the finalize queue — the session pushes
  /// newly-encoded windows here when entering the finalize-decode
  /// stage, and drains the queue front-to-back as decodes succeed.
  pub(super) fn finalize_queue_mut(&mut self) -> &mut VecDeque<PendingFinalize> {
    &mut self.finalize_queue
  }

  /// Push a freshly-encoded window onto the finalize queue.
  pub(super) fn enqueue_finalize(&mut self, window: Array) {
    self.finalize_queue.push_back(PendingFinalize {
      encoder_output: window,
      fallback_consumed: false,
    });
  }

  /// Discharge the [`RetryStage::StopEncoderFeed`] obligation, if any,
  /// against `encoder`. Returns the number of full windows committed
  /// by the staged `encoder.feed` (`0` on a sub-window staged buffer,
  /// `>= 1` on a window-completing one).
  ///
  /// Transactional: on Err the staged mel stays in `resume_at`. On Ok
  /// the resume point advances to [`RetryStage::DecodeOwed`] iff the
  /// drain produced one or more windows (so the next stage in the same
  /// call OR the next call's discharge runs the decode), or clears if
  /// `0` windows resulted (R5 corner: a sub-window drain owes no decode).
  ///
  /// Returns `Ok(0)` when there is no `StopEncoderFeed` obligation.
  pub(super) fn discharge_stop_encoder_feed<B>(
    &mut self,
    encoder: &mut StreamingEncoder<B>,
  ) -> Result<usize>
  where
    B: StreamingEncoderBackend,
  {
    let Some(RetryStage::StopEncoderFeed { mel_frames }) = self.resume_at.take() else {
      // Not our obligation — restore (we took() it above) and exit.
      // The take() only matched on StopEncoderFeed, so this branch is
      // unreachable, but the explicit check guards against future
      // refactors that might add another variant taking this path.
      return Ok(0);
    };
    // Run the fallible feed against the staged mel. If it errs, we
    // MUST restore the resume_at to its pre-call state so the next
    // call retries the SAME staged mel.
    let count = match encoder.feed(&mel_frames) {
      Ok(n) => n,
      Err(e) => {
        // ROLLBACK: re-arm the resume point. `mel_frames` was MOVED
        // into the match arm — restore by re-constructing.
        self.resume_at = Some(RetryStage::StopEncoderFeed { mel_frames });
        return Err(e);
      }
    };
    // COMMIT: resume_at already cleared by the take() above. If the
    // drain committed one or more windows, advance to DecodeOwed so
    // they flow through a decode pass BEFORE the next call returns.
    // Pre-rewrite this was a separate flag + a per-call local.
    if count > 0 {
      self.resume_at = Some(RetryStage::DecodeOwed);
    }
    Ok(count)
  }

  /// Stage a fresh `StopEncoderFeed` obligation — called by `stop()`
  /// after `mel.flush()` succeeds but BEFORE the encoder.feed call. If
  /// the feed errors, the resume point is already correct; on success
  /// the caller advances by clearing or chaining via the methods above.
  pub(super) fn stage_stop_encoder_feed(&mut self, mel_frames: Array) {
    self.resume_at = Some(RetryStage::StopEncoderFeed { mel_frames });
  }

  /// Stage a fresh `MelFlush` obligation — called by `stop()` BEFORE
  /// invoking `mel.flush()`. The mel processor's transactional `flush`
  /// preserves its `overlap_buffer` on Err, so the next call repeats
  /// the SAME flush.
  pub(super) fn stage_stop_mel_flush(&mut self) {
    self.resume_at = Some(RetryStage::StopMelFlush);
  }

  /// Clear the `MelFlush` obligation after a successful flush — called
  /// by `stop()`'s in-call commit.
  pub(super) fn clear_stop_mel_flush(&mut self) {
    if matches!(self.resume_at, Some(RetryStage::StopMelFlush)) {
      self.resume_at = None;
    }
  }

  /// Discharge the [`RetryStage::StopMelFlush`] obligation, if any,
  /// against `mel_processor`. Re-attempts the `flush()` whose previous
  /// invocation errored, and on success advances the resume point to
  /// [`RetryStage::StopEncoderFeed`] when the flush produced mel rows
  /// (so the next discharge step can drive the encoder feed).
  ///
  /// Returns the freshly-flushed `Option<Array>` so the caller can
  /// inspect it (the in-tree dispatcher discards it and falls through
  /// to [`discharge_stop_encoder_feed`](Self::discharge_stop_encoder_feed),
  /// but callers writing custom orchestrations can use it directly).
  ///
  /// Returns `Ok(None)` when there is no `StopMelFlush` obligation.
  ///
  /// # Transactional rollback
  /// - `mel_processor.flush()` Err → re-arms `StopMelFlush` so the next
  ///   call retries the SAME flush. `IncrementalMelSpectrogram::flush`
  ///   preserves `overlap_buffer` on Err (its own transactional
  ///   contract), so the retry sees identical input.
  /// - `Array::try_clone` on the flushed mel Err (rare — refcount-clone
  ///   only allocates a fresh handle slot) → re-arms `StopMelFlush` so
  ///   the next call retries the ENTIRE flush+stage. The freshly
  ///   flushed mel cannot be carried forward because its only handle
  ///   was needed both for the return value AND the `StopEncoderFeed`
  ///   stage, so we redo the whole step.
  ///
  /// # Errors
  /// Propagates from [`IncrementalMelSpectrogram::flush`] or from
  /// [`Array::try_clone`].
  pub(super) fn discharge_stop_mel_flush(
    &mut self,
    mel_processor: &mut IncrementalMelSpectrogram,
  ) -> Result<Option<Array>> {
    let Some(RetryStage::StopMelFlush) = self.resume_at else {
      return Ok(None);
    };

    // Take the obligation so we can either commit or re-arm.
    self.resume_at = None;

    match mel_processor.flush() {
      Ok(mel_opt) => {
        // Success: advance — if flush produced mel, stage StopEncoderFeed next.
        if let Some(mel) = mel_opt.as_ref() {
          match mel.try_clone() {
            Ok(cloned) => {
              self.resume_at = Some(RetryStage::StopEncoderFeed { mel_frames: cloned });
            }
            Err(e) => {
              // try_clone failed: re-arm StopMelFlush to retry the whole
              // flush+stage. The original mel handle (in `mel_opt`) is
              // dropped on the Err return — we cannot stage it because
              // the next discharge cannot recompute it from the (now-
              // cleared) overlap. The next StopMelFlush retry will
              // observe an empty overlap and short-circuit `Ok(None)`
              // unless new audio is fed first — which is exactly the
              // honest surface: the caller sees the Err and decides.
              self.resume_at = Some(RetryStage::StopMelFlush);
              return Err(Error::Backend {
                message: format!(
                  "StopMelFlush: failed to clone flushed mel for next-stage retry: {e}"
                ),
              });
            }
          }
        }
        Ok(mel_opt)
      }
      Err(e) => {
        // Rollback: re-arm StopMelFlush so next stop() retries.
        self.resume_at = Some(RetryStage::StopMelFlush);
        Err(e)
      }
    }
  }

  /// Mark that the same-call decode for one or more bridge-drained
  /// windows is OWED across call boundaries — called when a later
  /// fallible step in `feed_audio` errors AFTER the bridge drain
  /// successfully committed `count >= 1` windows to the encoder. The
  /// session's local count is lost to the `?` unwind; this flag is
  /// the cross-call source of truth (R6 corner).
  pub(super) fn arm_decode_owed(&mut self) {
    self.resume_at = Some(RetryStage::DecodeOwed);
  }

  /// Clear the `DecodeOwed` obligation after a successful decode pass.
  pub(super) fn clear_decode_owed(&mut self) {
    if matches!(self.resume_at, Some(RetryStage::DecodeOwed)) {
      self.resume_at = None;
    }
  }

  /// Mark that `stop()`'s post-finalize partial-window decode errored.
  /// The audio_features payload is carried in the stage so the retry
  /// doesn't have to recompute `encode_pending` — though the recompute
  /// would be safe (`encode_pending` is `&self` + idempotent), skipping
  /// it avoids a redundant encoder forward pass.
  pub(super) fn arm_stop_partial_decode(&mut self, audio_features: Option<Array>) {
    self.resume_at = Some(RetryStage::StopPartialDecode { audio_features });
  }

  /// True iff `resume_at == Some(StopPartialDecode)`.
  pub(super) fn has_pending_stop_partial_decode(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::StopPartialDecode { .. }))
  }

  /// Take the staged `StopPartialDecode` audio_features out of the
  /// resume point — used by `stop()`'s discharge to consume the
  /// payload while the resume point is being advanced. Returns `None`
  /// if `resume_at` doesn't currently hold a `StopPartialDecode`.
  pub(super) fn take_stop_partial_decode_features(&mut self) -> Option<Option<Array>> {
    if matches!(self.resume_at, Some(RetryStage::StopPartialDecode { .. })) {
      let Some(RetryStage::StopPartialDecode { audio_features }) = self.resume_at.take() else {
        unreachable!("matches! gated the take()")
      };
      Some(audio_features)
    } else {
      None
    }
  }

  /// Reset on cancel() / reset() — clears all obligations atomically.
  pub(super) fn clear_all(&mut self) {
    self.resume_at = None;
    self.finalize_queue.clear();
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Array;

  fn dummy_array() -> Array {
    Array::from_slice::<f32>(&[0.0_f32], &[1i32]).unwrap()
  }

  #[test]
  fn new_has_no_obligation() {
    let s = SessionRetryState::new();
    assert!(!s.has_obligation());
    assert!(s.resume_at().is_none());
    assert!(s.finalize_queue().is_empty());
  }

  #[test]
  fn enqueue_finalize_creates_obligation() {
    let mut s = SessionRetryState::new();
    s.enqueue_finalize(dummy_array());
    assert!(s.has_obligation());
    assert_eq!(s.finalize_queue().len(), 1);
    assert!(!s.finalize_queue()[0].fallback_consumed);
  }

  #[test]
  fn stage_stop_encoder_feed_then_clear_all_clears() {
    let mut s = SessionRetryState::new();
    s.stage_stop_encoder_feed(dummy_array());
    assert!(s.has_pending_stop_encoder_feed());
    s.clear_all();
    assert!(!s.has_pending_stop_encoder_feed());
    assert!(!s.has_obligation());
  }

  #[test]
  fn clear_all_drops_every_obligation_in_one_call() {
    let mut s = SessionRetryState::new();
    s.enqueue_finalize(dummy_array());
    s.arm_decode_owed();
    assert!(s.has_obligation());
    s.clear_all();
    assert!(!s.has_obligation());
    assert!(s.finalize_queue().is_empty());
    assert!(s.resume_at().is_none());
  }

  #[test]
  fn decode_owed_is_distinct_from_throttled_drain() {
    // R7 corner structural fix: there's no flag that bleeds across
    // calls when the same call's cadence throttle skipped the decode.
    // arm_decode_owed is the ONLY way to set DecodeOwed; the session's
    // happy-path drain (count > 0 + same-call decode succeeds) calls
    // clear_decode_owed AFTER the decode pass returns Ok and never
    // calls arm_decode_owed in the first place when the cadence
    // throttle declines the decode.
    let mut s = SessionRetryState::new();
    assert!(!s.has_decode_owed());
    s.arm_decode_owed();
    assert!(s.has_decode_owed());
    s.clear_decode_owed();
    assert!(!s.has_decode_owed());
  }

  #[test]
  fn take_stop_partial_decode_features_returns_none_when_not_set() {
    let mut s = SessionRetryState::new();
    assert!(s.take_stop_partial_decode_features().is_none());
  }

  #[test]
  fn take_stop_partial_decode_features_consumes_payload() {
    let mut s = SessionRetryState::new();
    s.arm_stop_partial_decode(Some(dummy_array()));
    assert!(s.has_pending_stop_partial_decode());
    let taken = s.take_stop_partial_decode_features().expect("set above");
    assert!(taken.is_some());
    assert!(!s.has_pending_stop_partial_decode());
  }

  // -------------------------------------------------------------------
  // F1: discharge_stop_mel_flush wiring + transactional contract.
  // -------------------------------------------------------------------

  /// F1: `discharge_stop_mel_flush` returns `Ok(None)` and clears the
  /// resume point when no obligation is set — the no-op short-circuit.
  #[test]
  fn discharge_stop_mel_flush_noop_when_not_staged() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    let out = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("noop must succeed");
    assert!(out.is_none(), "no obligation ⇒ Ok(None)");
    assert!(!s.has_obligation());
  }

  /// F1: with an empty overlap, `discharge_stop_mel_flush` clears the
  /// obligation and returns `Ok(None)` (mel.flush short-circuits on
  /// empty overlap). The resume point advances to `None`, NOT to
  /// `StopEncoderFeed` (no mel rows to stage).
  #[test]
  fn discharge_stop_mel_flush_empty_overlap_clears_obligation() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    s.stage_stop_mel_flush();
    assert!(s.has_pending_stop_mel_flush());

    let out = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("empty-overlap flush must succeed");
    assert!(
      out.is_none(),
      "empty overlap ⇒ flush yields None ⇒ no StopEncoderFeed stage"
    );
    assert!(!s.has_pending_stop_mel_flush());
    assert!(
      !s.has_pending_stop_encoder_feed(),
      "F1: no mel rows ⇒ MUST NOT advance to StopEncoderFeed"
    );
    assert!(!s.has_obligation());
  }

  /// F1: with non-empty overlap, `discharge_stop_mel_flush` runs the
  /// flush, advances `resume_at` to `StopEncoderFeed` carrying the
  /// fresh mel, and returns the flushed `Some(mel)` to the caller.
  /// The mel processor's overlap is cleared by the successful flush
  /// (the transactional contract — not the discharge's
  /// responsibility, but observable here).
  #[test]
  fn discharge_stop_mel_flush_with_overlap_advances_to_stop_encoder_feed() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    // Feed a chunk that's smaller than n_fft so process() returns None
    // and the samples accumulate in the overlap.
    let _ = mel
      .process(&[0.1_f32; 16])
      .expect("process must succeed on small input");
    assert!(
      mel.overlap_buffer_len() > 0,
      "test precondition: overlap populated"
    );

    s.stage_stop_mel_flush();
    let out = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("flush must succeed");
    assert!(out.is_some(), "non-empty overlap ⇒ flush yields Some(mel)");
    assert!(
      s.has_pending_stop_encoder_feed(),
      "F1: successful flush with mel rows MUST advance resume_at to \
       StopEncoderFeed for the downstream discharge to drain"
    );
    assert!(
      !s.has_pending_stop_mel_flush(),
      "F1: successful flush MUST clear StopMelFlush"
    );
  }
}
