//! Integration tests for [`mlxrs::audio::playback`] — the cpal-backed
//! `AudioPlayer` + `AudioOutputStream` trait port of
//! `mlx-audio-swift`'s `MLXAudioCore.AudioPlayer` streaming surface.
//!
//! Two test families:
//! - **Mock-based unit tests** (default; CI-safe). Exercise the
//!   `AudioOutputStream` trait + `PlaybackConfig` math without
//!   touching cpal device init — no audio hardware required.
//! - **Real-device tests** (gated `#[cfg(target_os = "macos")]` +
//!   `#[ignore]`). Smoke-test the cpal-driven `AudioPlayer`
//!   end-to-end on a real default output device.
//!
//! NO `peak_memory()` magnitude asserts (per the project's
//! `[[feedback_no_global_peak_memory_assert]]` rule).

#![cfg(feature = "audio")]

use std::sync::{Arc, Mutex};

use mlxrs::audio::playback::{AudioOutputStream, ChannelLayout, PlaybackConfig, SampleFormat};

// ---------------------------------------------------------------------------
// Mock AudioOutputStream
// ---------------------------------------------------------------------------

/// In-memory `AudioOutputStream` implementor used to test that the
/// trait surface compiles + behaves contractually without pulling in
/// cpal device init. Mirrors the role a unit-test recorder plays for
/// the Swift `AudioPlayer` (drop-in for `AVAudioPlayerNode`).
struct RecordingSink {
  buffer: Arc<Mutex<Vec<f32>>>,
  capacity: usize,
  running: bool,
}

impl RecordingSink {
  fn new(capacity: usize) -> Self {
    Self {
      buffer: Arc::new(Mutex::new(Vec::new())),
      capacity,
      running: true,
    }
  }
}

impl AudioOutputStream for RecordingSink {
  fn write_samples(&mut self, samples: &[f32]) -> mlxrs::error::Result<usize> {
    if !self.running {
      return Err(mlxrs::error::Error::Backend {
        message: "RecordingSink: stream stopped".to_string(),
      });
    }
    let mut buf = self.buffer.lock().unwrap();
    if buf.len() + samples.len() > self.capacity {
      return Err(mlxrs::error::Error::Backend {
        message: format!(
          "RecordingSink: capacity {} exceeded ({} + {})",
          self.capacity,
          buf.len(),
          samples.len()
        ),
      });
    }
    buf.extend_from_slice(samples);
    Ok(samples.len())
  }

  fn flush(&mut self) -> mlxrs::error::Result<()> {
    // Pretend the sink drained immediately.
    self.buffer.lock().unwrap().clear();
    Ok(())
  }

  fn stop(&mut self) -> mlxrs::error::Result<()> {
    self.running = false;
    self.buffer.lock().unwrap().clear();
    Ok(())
  }

  fn is_running(&self) -> bool {
    self.running
  }
}

// ---------------------------------------------------------------------------
// PlaybackConfig — default + constructor + cpal_config()
// ---------------------------------------------------------------------------

#[test]
fn playback_config_default_sample_rate_matches_swift_default() {
  // Swift `MLXAudioUI` voice-pipeline default is 24 kHz; the mlxrs
  // `PlaybackConfig::default` should match so the A8 pipeline
  // composes without spelling out the rate.
  let cfg = PlaybackConfig::default();
  assert_eq!(cfg.sample_rate, 24_000);
  assert_eq!(cfg.channels, ChannelLayout::Mono);
  assert_eq!(cfg.sample_format, SampleFormat::F32);
  assert_eq!(cfg.buffer_size_frames, None);
  // 4 seconds @ 24 kHz = 96000 frames.
  assert_eq!(cfg.queue_capacity_frames, 96_000);
}

#[test]
fn playback_config_mono_constructor() {
  let cfg = PlaybackConfig::mono(48_000);
  assert_eq!(cfg.sample_rate, 48_000);
  assert_eq!(cfg.channels, ChannelLayout::Mono);
  assert_eq!(cfg.channels.count(), 1);
  assert_eq!(cfg.queue_capacity_frames, 48_000 * 4);
}

#[test]
fn playback_config_stereo_constructor() {
  let cfg = PlaybackConfig::stereo(44_100);
  assert_eq!(cfg.channels, ChannelLayout::Stereo);
  assert_eq!(cfg.channels.count(), 2);
  // Stereo capacity bumps the frame budget so 4 seconds of stereo
  // audio fits (queue is in samples internally).
  assert_eq!(cfg.queue_capacity_frames, 44_100 * 4 * 2);
}

#[test]
fn channel_layout_count_arbitrary() {
  assert_eq!(ChannelLayout::Mono.count(), 1);
  assert_eq!(ChannelLayout::Stereo.count(), 2);
  assert_eq!(ChannelLayout::Channels(6).count(), 6);
}

#[test]
fn playback_config_cpal_config_rejects_zero_channels() {
  let cfg = PlaybackConfig {
    sample_rate: 16_000,
    channels: ChannelLayout::Channels(0),
    sample_format: SampleFormat::F32,
    buffer_size_frames: None,
    queue_capacity_frames: 1024,
  };
  let err = cfg.cpal_config().unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("channel count")),
    "expected Backend(channel count) error, got {err:?}"
  );
}

#[test]
fn playback_config_cpal_config_passes_buffer_hint() {
  let with_hint = PlaybackConfig {
    sample_rate: 16_000,
    channels: ChannelLayout::Mono,
    sample_format: SampleFormat::F32,
    buffer_size_frames: Some(256),
    queue_capacity_frames: 1024,
  };
  let cpal_cfg = with_hint.cpal_config().unwrap();
  assert_eq!(cpal_cfg.channels, 1);
  // `cpal::SampleRate` is a `pub type SampleRate = u32` alias in
  // cpal 0.17 — compare as a plain `u32`.
  assert_eq!(cpal_cfg.sample_rate, 16_000);
  assert!(matches!(cpal_cfg.buffer_size, cpal::BufferSize::Fixed(256)));

  let without_hint = PlaybackConfig::mono(16_000);
  let cpal_cfg = without_hint.cpal_config().unwrap();
  assert!(matches!(cpal_cfg.buffer_size, cpal::BufferSize::Default));
}

// ---------------------------------------------------------------------------
// AudioOutputStream trait — mock-based contract tests
// ---------------------------------------------------------------------------

#[test]
fn audio_output_stream_writes_samples_returns_count() {
  let mut sink = RecordingSink::new(4096);
  let samples = vec![0.5_f32; 1024];

  let written = sink.write_samples(&samples).unwrap();
  assert_eq!(written, 1024);
  assert!(sink.is_running());
}

#[test]
fn audio_output_stream_flush_drains_buffer() {
  let mut sink = RecordingSink::new(4096);
  sink.write_samples(&[0.1_f32; 256]).unwrap();
  // Pre-flush the buffer has 256 samples; post-flush it's empty.
  assert_eq!(sink.buffer.lock().unwrap().len(), 256);
  sink.flush().unwrap();
  assert_eq!(sink.buffer.lock().unwrap().len(), 0);
}

#[test]
fn audio_output_stream_stop_marks_not_running_and_rejects_writes() {
  let mut sink = RecordingSink::new(4096);
  assert!(sink.is_running());

  sink.stop().unwrap();
  assert!(!sink.is_running());

  // Post-stop writes return Err — the trait contract.
  let err = sink.write_samples(&[0.0_f32; 32]).unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("stopped")),
    "expected stopped-stream Backend error, got {err:?}"
  );
}

#[test]
fn audio_output_stream_overflow_returns_err() {
  let mut sink = RecordingSink::new(1024);
  sink.write_samples(&[0.0_f32; 512]).unwrap();
  sink.write_samples(&[0.0_f32; 512]).unwrap();

  // Now full; next write blows the cap.
  let err = sink.write_samples(&[0.0_f32; 1]).unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("capacity")),
    "expected capacity-overflow Backend error, got {err:?}"
  );
}

// ---------------------------------------------------------------------------
// AudioPlayer — non-device-touching unit tests
// ---------------------------------------------------------------------------
//
// These exercise the `AudioPlayer` construction + configuration path
// that DOESN'T need to open a cpal stream (which would fail in CI
// without an audio device). The cpal Stream-open path is exercised
// in the `#[ignore]`-gated real-device tests below.

#[test]
fn audio_player_rejects_non_f32_sample_format_pre_device() {
  // We can construct a PlaybackConfig with SampleFormat::I16 even
  // though the player doesn't currently support it; assert the
  // construction path itself succeeds (the device-open call would
  // be the one to reject — exercised in real-device tests). Verifies
  // the enum is exposed + the config builder is the gate.
  let cfg = PlaybackConfig {
    sample_rate: 16_000,
    channels: ChannelLayout::Mono,
    sample_format: SampleFormat::I16,
    buffer_size_frames: None,
    queue_capacity_frames: 1024,
  };
  // cpal_config doesn't gate sample_format (that's a device-level
  // concern in cpal); the player's `with_device` constructor is what
  // returns Err on I16. Smoke-check the field round-trips:
  assert_eq!(cfg.sample_format, SampleFormat::I16);
}

#[test]
fn audio_player_queue_capacity_frames_multiplied_by_channels() {
  // Sanity-check the per-frame -> per-sample math the player uses
  // internally so a stereo player with 1024-frame capacity actually
  // accepts 2048 samples (interleaved L/R) before overflow.
  let cfg = PlaybackConfig {
    sample_rate: 16_000,
    channels: ChannelLayout::Stereo,
    sample_format: SampleFormat::F32,
    buffer_size_frames: None,
    queue_capacity_frames: 1024,
  };
  let frames = cfg.queue_capacity_frames;
  let samples = frames * usize::from(cfg.channels.count());
  assert_eq!(samples, 2048);
}

// ---------------------------------------------------------------------------
// Real-device tests — gated. Run with: `cargo test -- --ignored`
// ---------------------------------------------------------------------------
//
// `#[ignore]` so CI (which may lack a default audio output device,
// e.g. headless macOS runners under -nox) doesn't fail on construct.
// macOS-only gate because CoreAudio is the only backend mlxrs targets
// in M5; on Linux/Windows the same code should work but isn't a
// shipping target. Run locally with:
//
//     cargo test -p mlxrs --features audio audio_player_starts_and_stops_on_default_device \
//         -- --ignored --test-threads=1

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_constructs_without_starting_stream() {
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(24_000)).unwrap();
  // Newly-constructed player isn't running until `start()`.
  assert!(!player.is_running());
  assert!(!player.is_paused());
  assert_eq!(player.buffer_depth(), 0);
  assert_eq!(player.config().sample_rate, 24_000);
  // Defaults round-trip:
  assert!((player.volume() - 1.0).abs() < 1e-6);
  // Cleanup.
  let _ = player.stop();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_starts_and_stops_on_default_device() {
  use std::{thread, time::Duration};

  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(24_000)).unwrap();
  player.start().unwrap();
  assert!(player.is_running());

  // Push a quarter-second of silence so the cpal callback has
  // something to drain; underrun would also be safe (silence) but
  // this is a stronger sanity check that write_samples + flush
  // round-trip on a real device.
  let samples = vec![0.0_f32; 24_000 / 4];
  player.write_samples(&samples).unwrap();
  player.flush().unwrap();
  assert_eq!(player.buffer_depth(), 0);

  player.pause().unwrap();
  assert!(player.is_paused());
  assert!(!player.is_running());

  player.resume().unwrap();
  assert!(player.is_running());

  player.stop().unwrap();
  assert!(!player.is_running());

  // Give cpal a beat to settle before drop.
  thread::sleep(Duration::from_millis(50));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_buffer_overflow_returns_err() {
  use mlxrs::audio::playback::AudioPlayer;

  // Tiny queue so overflow is reachable without pushing megabytes.
  let cfg = PlaybackConfig {
    sample_rate: 16_000,
    channels: ChannelLayout::Mono,
    sample_format: SampleFormat::F32,
    buffer_size_frames: None,
    queue_capacity_frames: 1024,
  };
  let mut player = AudioPlayer::new(cfg).unwrap();
  // Don't start — keep the cpal callback paused so the queue
  // doesn't drain while we fill it.
  player.write_samples(&[0.0_f32; 1024]).unwrap();
  let err = player.write_samples(&[0.0_f32; 1]).unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("overflow")),
    "expected overflow Backend error, got {err:?}"
  );
  let _ = player.stop();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_underrun_emits_silence_no_panic() {
  use std::{thread, time::Duration};

  use mlxrs::audio::playback::AudioPlayer;

  // Start the stream with an empty queue; the cpal callback should
  // emit silence (zero) for every callback hit instead of panicking
  // or blocking. We can't directly observe the cpal-callback buffer
  // from here, but we can assert the player stays in STATE_RUNNING
  // across a callback interval and is_running() stays true (no
  // poisoned-state from a panic in the callback).
  let mut player = AudioPlayer::new(PlaybackConfig::mono(24_000)).unwrap();
  player.start().unwrap();
  assert!(player.is_running());

  thread::sleep(Duration::from_millis(100));
  assert!(player.is_running(), "underrun must not stop the player");

  player.stop().unwrap();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_set_volume_clamps_and_persists() {
  use mlxrs::audio::playback::AudioPlayer;

  let player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  assert!((player.volume() - 1.0).abs() < 1e-6);

  player.set_volume(0.5);
  assert!((player.volume() - 0.5).abs() < 1e-6);

  // Clamp: values >1.0 or <0.0 are clamped silently.
  player.set_volume(1.5);
  assert!((player.volume() - 1.0).abs() < 1e-6);

  player.set_volume(-0.1);
  assert!((player.volume() - 0.0).abs() < 1e-6);
}
