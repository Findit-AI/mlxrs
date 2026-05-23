//! [`AudioPlayer`] — cpal-backed device playback, the mlxrs port of
//! `mlx-audio-swift`'s
//! [`MLXAudioCore.AudioPlayer.startStreaming(sampleRate:)`][swift-ap] +
//! `scheduleAudioChunk(_:withCrossfade:)` streaming path.
//!
//! ## API mirror
//!
//! Swift `AudioPlayer` exposes a roughly six-method streaming surface
//! over `AVAudioEngine` + `AVAudioPlayerNode`:
//!
//! | Swift                                       | mlxrs                                              |
//! | ------------------------------------------- | -------------------------------------------------- |
//! | `startStreaming(sampleRate:)`               | [`AudioPlayer::new`] / [`AudioPlayer::with_device`] + [`AudioPlayer::start`] |
//! | `scheduleAudioChunk(_:withCrossfade:)`      | [`AudioPlayer::write_samples`] (via [`super::output_stream::AudioOutputStream`]) |
//! | `pause()` (streaming branch)                | [`AudioPlayer::pause`]                              |
//! | `togglePlayPause()` (streaming branch)      | [`AudioPlayer::resume`]                             |
//! | `stopStreaming()` / `stop()`                | [`AudioPlayer::stop`] / `Drop`                      |
//! | `isPlaying` / `isStreamingMode`             | [`AudioPlayer::is_running`]                         |
//! | `finishStreamingInput()`                    | [`super::output_stream::AudioOutputStream::flush`]  |
//!
//! Swift's volume is read off `AVAudioPlayerNode.volume`; mlxrs
//! exposes the equivalent via [`AudioPlayer::set_volume`] +
//! [`AudioPlayer::volume`] backed by an `AtomicU32` (f32 bits) the
//! cpal callback reads each invocation.
//!
//! ## Cpal callback + buffer-queue plumbing
//!
//! The Swift path is buffer-by-buffer (`AVAudioPlayerNode.scheduleBuffer`
//! per `[Float]` chunk, each carrying its own completion handler).
//! The cpal equivalent inverts the polarity: cpal owns the I/O
//! thread, calls back into us with a pre-sized `&mut [f32]` to fill,
//! and we pull samples from a thread-safe queue. Concretely:
//!
//! ```text
//! producer thread (e.g. STS pipeline)            cpal I/O thread
//! ───────────────────────────────────            ─────────────────
//!   write_samples(&[f32]) ──┐                          │
//!                           ▼                          │
//!                 SampleQueue::push                    │
//!                           │                          │
//!                           ▼                          ▼
//!                       Arc<Mutex<VecDeque<f32>>> ── callback fills &mut [f32]
//!                           │                          │
//!                           │                          ▼
//!                           │              for s in out: s = pop_or_zero() * volume
//!                           ▼
//!                  AudioPlayer::buffer_depth
//! ```
//!
//! - **Producer side.** [`AudioPlayer::write_samples`] locks the
//!   shared `VecDeque<f32>` (capped at
//!   [`super::config::PlaybackConfig::queue_capacity_frames`] × channel
//!   count). Returns `Err` on overflow (a recoverable
//!   [`crate::error::Error::Backend`] — no producer surprise OOM).
//!   This is the cpal-equivalent of `AVAudioPlayerNode.scheduleBuffer`
//!   returning even though the underlying scheduling chain is
//!   bounded.
//! - **Cpal callback.** Runs on cpal's audio I/O thread; locks the
//!   queue (a short critical section — only `pop_front` calls under
//!   the lock), reads the current volume from the `AtomicU32`,
//!   writes `pop * volume` per sample. On underrun (queue empty)
//!   the callback writes `0.0` — silence — instead of panicking or
//!   blocking. This matches the Swift behavior: the player node
//!   sits idle if no buffer is scheduled.
//! - **State.** Stored as `Arc<AtomicU8>` so both producer and cpal
//!   callback can observe transitions without holding the queue
//!   lock; values [`STATE_STOPPED`], [`STATE_RUNNING`],
//!   [`STATE_PAUSED`] map to Swift's `isPlaying` /
//!   `isStreamingMode` distinction (we collapse them into a single
//!   tri-state to make the cpal-side check a single atomic load).
//!
//! ## Scope cuts (explicit, A11)
//!
//! The Swift `AudioPlayer` exposes a few capabilities A11 deliberately
//! does NOT port; each is a separate follow-up issue per the
//! `[[feedback_match_official_binding_design]]` rule:
//!
//! - **Audio input / recording.** A11 is playback-only.
//!   `AVAudioPlayer` / `AVAudioPlayerDelegate` are not mirrored.
//! - **File I/O (`loadAudio(from: URL)`).** A11 plays raw PCM. WAV /
//!   MP3 / FLAC loading already lives in [`crate::audio::io`]; a
//!   caller that wants to play a file decodes there and pipes the
//!   resulting samples through [`AudioPlayer::write_samples`].
//! - **Format conversion (`PCMStreamConverter`).** A11 expects the
//!   caller to supply samples at the configured
//!   [`super::config::PlaybackConfig::sample_rate`] /
//!   [`super::config::PlaybackConfig::channels`]. Resampling +
//!   format-conversion is a separate concern (already partially
//!   covered by [`crate::audio::io::load_audio`]'s resampling, fully
//!   covered by a future polyphase resampler follow-up).
//! - **Crossfade / fade-in (`scheduleAudioChunk(_:withCrossfade:)`'s
//!   `withCrossfade: true` branch).** Crossfade is an
//!   application-level concern; A11 plays exactly the samples the
//!   caller pushes. A future helper module can wrap `AudioPlayer`
//!   with a fade-in/crossfade transform without touching the
//!   playback core.
//! - **Per-buffer completion callbacks.** Swift schedules each
//!   buffer with a `completionCallbackType: .dataConsumed` to track
//!   queued-buffer drain; cpal has no per-buffer-completion hook —
//!   instead [`super::output_stream::AudioOutputStream::flush`]
//!   blocks until [`AudioPlayer::buffer_depth`] reaches zero, which
//!   is the same end-state contract (`onDidFinishStreaming` fires
//!   when `queuedBuffers == 0`).
//! - **Timer-driven `currentTime` publishing.** Swift uses
//!   `Timer.scheduledTimer` (every 100ms) + Combine to publish
//!   `currentTime` for UI binding. mlxrs is a Rust library, not a
//!   SwiftUI ObservableObject; no `@Published` properties / no
//!   Combine equivalent. Callers that want positional readback can
//!   maintain their own sample counter against
//!   [`AudioPlayer::buffer_depth`].
//!
//! [swift-ap]: https://github.com/fintit-ai/mlx-audio-swift/blob/main/Sources/MLXAudioCore/AudioPlayer.swift

use std::{
  collections::VecDeque,
  sync::{
    Arc, Mutex,
    atomic::{AtomicU8, AtomicU32, Ordering},
  },
  thread,
  time::{Duration, Instant},
};

use cpal::{
  Stream, StreamError,
  traits::{DeviceTrait, HostTrait, StreamTrait},
};

use super::{
  config::{PlaybackConfig, SampleFormat},
  output_stream::AudioOutputStream,
};
use crate::error::{Error, Result};

/// Stopped — the cpal stream is built but not playing (Swift's
/// `!isStreaming && !isPlaying`).
pub const STATE_STOPPED: u8 = 0;
/// Running — the cpal stream is `play()`ing and producer writes are
/// accepted (Swift's `isStreaming && isPlaying`).
pub const STATE_RUNNING: u8 = 1;
/// Paused — the cpal stream is `pause()`d but the queue retains its
/// contents (Swift's `playerNode.pause()` branch of `pause()`).
pub const STATE_PAUSED: u8 = 2;

/// Spin-wait granularity for [`AudioPlayer::flush`]. Picked to match
/// Swift's `Timer.scheduledTimer(withTimeInterval: 0.1, ...)` poll
/// cadence so flush latency under tight contention is bounded by the
/// same order of magnitude as the Swift implementation.
const FLUSH_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Default [`AudioPlayer::flush`] timeout. Defensive cap so a stalled
/// cpal device doesn't block the producer forever; long enough that a
/// realistic 4-second queue (the [`PlaybackConfig`] default) can drain
/// at real-time playback speeds with safety margin.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);

/// Thread-shared callback context. Lives behind an `Arc` so the cpal
/// stream's callback (which gets a `'static` closure) and the
/// producer-side [`AudioPlayer`] can both read/write the same state.
///
/// Kept as a dedicated struct (rather than five sibling `Arc<…>`
/// fields on `AudioPlayer`) so the `Drop` impl on [`AudioPlayer`]
/// can drop the cpal stream first (which joins the callback thread)
/// without an interleaved-drop hazard on the queue / state /
/// volume atomics.
struct SharedState {
  /// Producer-consumer queue of interleaved f32 samples. Bounded at
  /// `PlaybackConfig::queue_capacity_frames * channels` total
  /// samples; the cap is enforced in [`AudioPlayer::write_samples`].
  ///
  /// `Mutex` (not `parking_lot::Mutex`, not lock-free `ringbuf`) is
  /// chosen for A11 because:
  /// - cpal's audio thread takes the lock for a single
  ///   `pop_front`-loop per callback (microseconds at typical 64-1024
  ///   frame callback buffers),
  /// - the producer holds the lock only across `extend` +
  ///   capacity-check arithmetic,
  /// - a future migration to `ringbuf` (one of the cpal docs'
  ///   recommended low-latency choices) is a local refactor behind
  ///   the same trait surface if profiling shows the lock matters.
  queue: Mutex<VecDeque<f32>>,
  /// Bound on `queue.lock().unwrap().len()`; computed once from
  /// `PlaybackConfig::queue_capacity_frames * channels.count()` so
  /// the producer doesn't recompute it per `write_samples` call.
  queue_capacity_samples: usize,
  /// Current state. Loaded by the cpal callback on every invocation
  /// (single atomic load is the lightweight check that gates the
  /// pop loop); written by the producer (`start`, `pause`, `resume`,
  /// `stop`).
  state: AtomicU8,
  /// Current volume scalar, stored as `f32::to_bits` in an
  /// `AtomicU32`. Read by the cpal callback every sample; written
  /// by [`AudioPlayer::set_volume`]. Default is 1.0 (unity gain) —
  /// matches Swift's `AVAudioPlayerNode.volume` default.
  volume_bits: AtomicU32,
  /// Captured first error from the cpal stream's `err_fn`. The
  /// callback can't bubble up `Result`, so we stash it here and
  /// surface it on the next producer call (`write_samples`, `flush`,
  /// `pause`, `resume`).
  ///
  /// `Mutex<Option<String>>` (string-typed, not `Error`-typed) so
  /// errors aren't lost if multiple device events fire — we keep the
  /// first one. Cleared by [`AudioPlayer::stop`].
  callback_error: Mutex<Option<String>>,
}

impl SharedState {
  fn new(queue_capacity_samples: usize) -> Self {
    Self {
      queue: Mutex::new(VecDeque::with_capacity(
        queue_capacity_samples.min(64 * 1024),
      )),
      queue_capacity_samples,
      state: AtomicU8::new(STATE_STOPPED),
      volume_bits: AtomicU32::new(1.0_f32.to_bits()),
      callback_error: Mutex::new(None),
    }
  }

  fn load_volume(&self) -> f32 {
    f32::from_bits(self.volume_bits.load(Ordering::Relaxed))
  }
}

/// Cpal-backed device player.
///
/// See the module-level docs for the cpal-callback + buffer-queue
/// plumbing diagram and the explicit list of Swift-side capabilities
/// A11 scopes out (input, file I/O, format conversion, crossfade,
/// per-buffer completions, `@Published` properties).
pub struct AudioPlayer {
  /// The cpal output stream. `None` only between
  /// [`AudioPlayer::stop`] + `Drop` (we tear the stream down on
  /// `stop` so we can rebuild a fresh one on a subsequent `start`,
  /// matching Swift's `startStreaming` ↔ `stopStreaming` lifecycle).
  ///
  /// The current implementation builds the stream once at
  /// construction time and re-uses it for the full lifetime of the
  /// player (cpal Streams support `play()` / `pause()`); we still
  /// keep this `Option` so `Drop` can take + drop it explicitly
  /// before the `SharedState` so the cpal callback thread is joined
  /// while the queue + atomics are still live.
  ///
  /// `cpal::Stream` is `Send + Sync` (per cpal 0.17.x docs) so the
  /// `AudioPlayer` can cross thread boundaries — the A8 pipeline can
  /// drive a player from any thread.
  stream: Option<Stream>,
  /// Shared callback + producer state. See [`SharedState`].
  shared: Arc<SharedState>,
  /// Stored config; consulted by [`AudioPlayer::config`] introspection
  /// + the [`AudioOutputStream`] impl.
  config: PlaybackConfig,
}

impl AudioPlayer {
  /// Build an [`AudioPlayer`] bound to the default output device on
  /// the default cpal host. Mirrors the Swift
  /// `AudioPlayer.startStreaming(sampleRate:)` entry point (which
  /// implicitly uses `AVAudioEngine`'s default output node).
  ///
  /// The cpal stream is **built but not started** — call
  /// [`AudioPlayer::start`] before pushing samples. This matches the
  /// Swift split between `startStreaming` (engine prep) and
  /// `playerNode.play()` (actual playback).
  ///
  /// # Errors
  /// - [`Error::Backend`] if cpal has no default host, no default
  ///   output device, the config rejects, or the cpal stream build
  ///   fails (CoreAudio init failure, unsupported sample rate,
  ///   etc.).
  pub fn new(config: PlaybackConfig) -> Result<Self> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| Error::Backend {
      message: "AudioPlayer: no default cpal output device available".to_string(),
    })?;
    Self::with_device(&device, config)
  }

  /// Build an [`AudioPlayer`] bound to an explicit cpal device.
  /// Useful when the caller has already enumerated cpal devices and
  /// wants to target a specific one (the Swift API has no direct
  /// analog — `AVAudioEngine` always uses the system default — but
  /// cpal's multi-device support is a natural extension here).
  ///
  /// # Errors
  /// - [`Error::Backend`] if [`PlaybackConfig::cpal_config`] rejects
  ///   the config (zero channels) or the cpal stream build fails.
  pub fn with_device(device: &cpal::Device, config: PlaybackConfig) -> Result<Self> {
    if !matches!(config.sample_format, SampleFormat::F32) {
      return Err(Error::Backend {
        message: format!(
          "AudioPlayer: only SampleFormat::F32 is currently supported (got {:?}); \
           non-F32 device negotiation is reserved for a follow-up",
          config.sample_format
        ),
      });
    }

    let stream_config = config.cpal_config()?;

    let queue_capacity_samples = config
      .queue_capacity_frames
      .checked_mul(usize::from(config.channels.count()))
      .ok_or_else(|| Error::Backend {
        message: "AudioPlayer: queue_capacity_frames * channels overflows usize".to_string(),
      })?;

    let shared = Arc::new(SharedState::new(queue_capacity_samples));

    // cpal callback (audio I/O thread). Pulls from the queue, scales
    // by current volume, writes silence on underrun. Cloned `Arc`
    // moved into the `'static` closure cpal requires.
    let cb_shared = Arc::clone(&shared);
    let data_callback = move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
      let state = cb_shared.state.load(Ordering::Acquire);
      if state != STATE_RUNNING {
        // Paused / stopped — emit silence. (Cpal pauses the
        // callback on `Stream::pause()`, but the producer may also
        // toggle our `state` flag; the dual gate is intentional.)
        for s in out.iter_mut() {
          *s = 0.0;
        }
        return;
      }
      let volume = cb_shared.load_volume();
      // Single short critical section: drain into the cpal buffer.
      // We don't hold the lock across `*s = ...` arithmetic outside
      // this scope.
      let mut q = match cb_shared.queue.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      let drain_n = out.len().min(q.len());
      for slot in out.iter_mut().take(drain_n) {
        // pop_front is O(1) for VecDeque; the loop is the cpal
        // equivalent of the Swift `AVAudioPCMBuffer` per-buffer copy.
        let sample = q.pop_front().unwrap_or(0.0);
        *slot = sample * volume;
      }
      // Drop the lock before zeroing the tail — silence-on-underrun
      // doesn't need the queue.
      drop(q);
      for slot in out.iter_mut().skip(drain_n) {
        *slot = 0.0;
      }
    };

    // cpal `err_fn`. Stash the first error; surface it on the next
    // producer call. We don't have a logger dep in mlxrs, so silent
    // capture is the chosen behavior (the producer will see it).
    let err_shared = Arc::clone(&shared);
    let err_callback = move |err: StreamError| {
      let mut slot = match err_shared.callback_error.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      if slot.is_none() {
        *slot = Some(format!("cpal stream error: {err}"));
      }
    };

    let stream = device
      .build_output_stream(&stream_config, data_callback, err_callback, None)
      .map_err(|e| Error::Backend {
        message: format!("AudioPlayer: cpal build_output_stream failed: {e}"),
      })?;

    Ok(Self {
      stream: Some(stream),
      shared,
      config,
    })
  }

  /// Read-only view of the config the player was built with.
  #[must_use]
  pub fn config(&self) -> &PlaybackConfig {
    &self.config
  }

  /// Number of samples currently queued for playback (the cpal
  /// equivalent of Swift's `queuedBuffers * buffer.frameLength` sum,
  /// in samples not frames).
  #[must_use]
  pub fn buffer_depth(&self) -> usize {
    match self.shared.queue.lock() {
      Ok(g) => g.len(),
      Err(poisoned) => poisoned.into_inner().len(),
    }
  }

  /// `true` if [`AudioPlayer::start`] has been called and neither
  /// [`AudioPlayer::pause`] nor [`AudioPlayer::stop`] has run since.
  /// Mirrors the Swift `isPlaying` getter on the streaming branch.
  #[must_use]
  pub fn is_running(&self) -> bool {
    self.shared.state.load(Ordering::Acquire) == STATE_RUNNING
  }

  /// `true` if the player is in [`STATE_PAUSED`] (cpal stream is
  /// `pause()`d, queue retains samples; Swift's `playerNode.pause()`
  /// branch of `AudioPlayer.pause()`).
  #[must_use]
  pub fn is_paused(&self) -> bool {
    self.shared.state.load(Ordering::Acquire) == STATE_PAUSED
  }

  /// Current output volume, default 1.0. Mirrors
  /// `AVAudioPlayerNode.volume`.
  #[must_use]
  pub fn volume(&self) -> f32 {
    self.shared.load_volume()
  }

  /// Set the output volume. Clamped to `[0.0, 1.0]` — values outside
  /// the range are clamped silently (matches the
  /// `AVAudioPlayerNode.volume` 0..1 documented range).
  ///
  /// Takes `&self` (not `&mut self`) so the volume can be adjusted
  /// concurrently with [`AudioPlayer::write_samples`] without
  /// shadowing the producer borrow — useful when a UI thread tweaks
  /// volume while a worker thread is pumping samples.
  pub fn set_volume(&self, vol: f32) {
    let clamped = vol.clamp(0.0, 1.0);
    self
      .shared
      .volume_bits
      .store(clamped.to_bits(), Ordering::Relaxed);
  }

  /// Start the cpal stream — samples written via
  /// [`AudioPlayer::write_samples`] start flowing to the device.
  /// Mirrors the Swift `playerNode.play()` call inside
  /// `startStreaming`.
  ///
  /// Idempotent: calling `start` on a running player is a no-op
  /// (returns `Ok(())`). Calling `start` after `pause` resumes
  /// playback (equivalent to [`AudioPlayer::resume`]).
  ///
  /// # Errors
  /// - [`Error::Backend`] if the cpal `Stream::play()` call fails,
  ///   or if the stream has already been dropped by a prior `stop`.
  pub fn start(&mut self) -> Result<()> {
    self.take_callback_error()?;
    let stream = self.stream.as_ref().ok_or_else(|| Error::Backend {
      message: "AudioPlayer::start: stream has been dropped (post-stop)".to_string(),
    })?;
    stream.play().map_err(|e| Error::Backend {
      message: format!("AudioPlayer::start: cpal play() failed: {e}"),
    })?;
    self.shared.state.store(STATE_RUNNING, Ordering::Release);
    Ok(())
  }

  /// Pause playback. The cpal stream is `pause()`d and the queue
  /// retains its samples; subsequent [`AudioPlayer::write_samples`]
  /// calls still buffer into the queue but no audio is emitted.
  /// Mirrors `MLXAudioCore.AudioPlayer.pause()` (streaming branch).
  ///
  /// # Errors
  /// - [`Error::Backend`] if the cpal `Stream::pause()` call fails.
  pub fn pause(&mut self) -> Result<()> {
    self.take_callback_error()?;
    let stream = self.stream.as_ref().ok_or_else(|| Error::Backend {
      message: "AudioPlayer::pause: stream has been dropped (post-stop)".to_string(),
    })?;
    stream.pause().map_err(|e| Error::Backend {
      message: format!("AudioPlayer::pause: cpal pause() failed: {e}"),
    })?;
    self.shared.state.store(STATE_PAUSED, Ordering::Release);
    Ok(())
  }

  /// Resume from [`AudioPlayer::pause`]. Mirrors Swift's
  /// `togglePlayPause()` resuming branch (`playerNode.play()` +
  /// `isPlaying = true`).
  ///
  /// # Errors
  /// - [`Error::Backend`] if the cpal `Stream::play()` call fails.
  pub fn resume(&mut self) -> Result<()> {
    self.start()
  }

  /// Stop playback immediately. Drops every queued sample, pauses
  /// the cpal stream (we don't tear it down so a subsequent
  /// [`AudioPlayer::start`] still works), and clears any captured
  /// callback error. Mirrors `stopStreaming()`.
  ///
  /// # Errors
  /// - [`Error::Backend`] if the cpal `Stream::pause()` call fails.
  pub fn stop(&mut self) -> Result<()> {
    self.shared.state.store(STATE_STOPPED, Ordering::Release);
    if let Some(stream) = self.stream.as_ref() {
      stream.pause().map_err(|e| Error::Backend {
        message: format!("AudioPlayer::stop: cpal pause() failed: {e}"),
      })?;
    }
    // Clear queue + callback-error.
    if let Ok(mut q) = self.shared.queue.lock() {
      q.clear();
    }
    if let Ok(mut e) = self.shared.callback_error.lock() {
      *e = None;
    }
    Ok(())
  }

  /// Push interleaved PCM samples into the playback queue. Returns
  /// the number of samples accepted (`= samples.len()` on success).
  ///
  /// Surfaces a pending callback error (cpal `err_fn` capture) if
  /// one is queued — the next producer call after a device error
  /// receives the error report.
  ///
  /// # Errors
  /// - [`Error::Backend`] if the queue would overflow
  ///   [`PlaybackConfig::queue_capacity_frames`] × channel count.
  ///   The write is rejected wholesale — no partial accept on
  ///   overflow (the caller has no way to know how many fit, and a
  ///   partial accept would invite torn audio at the chunk
  ///   boundary).
  /// - [`Error::Backend`] if a prior cpal callback error was
  ///   captured.
  pub fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
    self.take_callback_error()?;

    let mut q = match self.shared.queue.lock() {
      Ok(g) => g,
      Err(poisoned) => poisoned.into_inner(),
    };
    let projected_len = q
      .len()
      .checked_add(samples.len())
      .ok_or_else(|| Error::Backend {
        message: "AudioPlayer::write_samples: queue length + new samples overflows usize"
          .to_string(),
      })?;
    if projected_len > self.shared.queue_capacity_samples {
      return Err(Error::Backend {
        message: format!(
          "AudioPlayer::write_samples: queue overflow (capacity {} samples, current {} samples, \
           tried to push {})",
          self.shared.queue_capacity_samples,
          q.len(),
          samples.len()
        ),
      });
    }
    q.extend(samples.iter().copied());
    Ok(samples.len())
  }

  /// Block until the playback queue has drained. The cpal callback
  /// continues to consume samples while this method polls; when the
  /// queue empties, [`AudioPlayer::flush`] returns. Mirrors Swift's
  /// `finishStreamingInput()` → `finishStreamIfDrained()` path.
  ///
  /// The implementation is a bounded poll loop (10ms granularity,
  /// 30s timeout) — cpal has no per-buffer-completion hook so we
  /// can't park on a condvar tied to the callback. The poll cadence
  /// matches Swift's `Timer.scheduledTimer(withTimeInterval: 0.1)`
  /// order of magnitude; the timeout prevents an indefinite block
  /// on a stalled device.
  ///
  /// If the player is not [`STATE_RUNNING`] (stopped or paused) and
  /// the queue is non-empty, this method returns immediately with a
  /// [`Error::Backend`] — flushing a stopped/paused player would
  /// block forever (the callback doesn't drain unless running).
  ///
  /// # Errors
  /// - [`Error::Backend`] if the flush times out, the player is not
  ///   running and the queue is non-empty, or a cpal callback error
  ///   surfaced mid-drain.
  pub fn flush(&mut self) -> Result<()> {
    self.take_callback_error()?;
    let start = Instant::now();
    loop {
      let depth = self.buffer_depth();
      if depth == 0 {
        return Ok(());
      }
      let state = self.shared.state.load(Ordering::Acquire);
      if state != STATE_RUNNING {
        return Err(Error::Backend {
          message: format!(
            "AudioPlayer::flush: queue has {depth} samples but state is {state} (not running) — \
             call start() before flush()"
          ),
        });
      }
      if start.elapsed() > FLUSH_TIMEOUT {
        return Err(Error::Backend {
          message: format!(
            "AudioPlayer::flush: timed out after {:?} with {depth} samples still queued",
            FLUSH_TIMEOUT
          ),
        });
      }
      thread::sleep(FLUSH_POLL_INTERVAL);
      self.take_callback_error()?;
    }
  }

  /// Pull the captured cpal `err_fn` message (if any) and surface it
  /// as a [`Error::Backend`]. Called at the head of every public
  /// producer method.
  fn take_callback_error(&self) -> Result<()> {
    let mut slot = match self.shared.callback_error.lock() {
      Ok(g) => g,
      Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(msg) = slot.take() {
      return Err(Error::Backend { message: msg });
    }
    Ok(())
  }
}

impl AudioOutputStream for AudioPlayer {
  fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
    AudioPlayer::write_samples(self, samples)
  }

  fn flush(&mut self) -> Result<()> {
    AudioPlayer::flush(self)
  }

  fn stop(&mut self) -> Result<()> {
    AudioPlayer::stop(self)
  }

  fn is_running(&self) -> bool {
    AudioPlayer::is_running(self)
  }
}

impl Drop for AudioPlayer {
  fn drop(&mut self) {
    // Mark stopped first so the callback sees STATE_STOPPED on its
    // next invocation and stops draining.
    self.shared.state.store(STATE_STOPPED, Ordering::Release);
    // Drop the stream explicitly. `cpal::Stream`'s `Drop` joins the
    // I/O thread (so the data callback is guaranteed dead after
    // this line); doing it explicitly + first means the callback
    // can't observe a half-dropped `SharedState`.
    if let Some(stream) = self.stream.take() {
      // Best-effort pause before drop — on macOS CoreAudio,
      // `Stream::drop` already stops the unit, but pausing first
      // avoids one extra callback hit on `STATE_STOPPED` silence.
      let _ = stream.pause();
      drop(stream);
    }
  }
}
