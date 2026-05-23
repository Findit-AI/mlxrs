//! Streaming speech-to-text — incremental encoder + orchestration.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/`][swift-dir]:
//!
//! - [`crate::audio::stt::streaming::mel_spectrogram::IncrementalMelSpectrogram`] —
//!   streaming mel-spec with overlap-save framing, mirrors
//!   `IncrementalMelSpectrogram.swift`.
//! - [`crate::audio::stt::streaming::encoder::StreamingEncoderBackend`]
//!   trait +
//!   [`crate::audio::stt::streaming::encoder::StreamingEncoder`] window
//!   accumulator — mirrors `StreamingEncoder.swift`.
//! - [`crate::audio::stt::streaming::types::DelayPreset`],
//!   [`crate::audio::stt::streaming::types::StreamingConfig`],
//!   [`crate::audio::stt::streaming::types::TranscriptionEvent`], and
//!   [`crate::audio::stt::streaming::types::StreamingStats`] —
//!   value-types, mirror `StreamingTypes.swift`.
//!
//! A subsequent commit adds `session` (`StreamingInferenceSession` +
//! `StreamingDecoderBackend` trait, mirroring
//! `StreamingInferenceSession.swift`).
//!
//! Per the project's [no per-model arch porting][noarch] rule, mlxrs
//! ships **no** concrete encoder / decoder implementations:
//! per-architecture encoders / decoders implement the streaming traits
//! and pass themselves into the streaming session.
//!
//! [swift-dir]: https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioSTT/Streaming
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

pub mod encoder;
pub mod mel_spectrogram;
pub mod types;

pub use encoder::{StreamingEncoder, StreamingEncoderBackend};
pub use mel_spectrogram::IncrementalMelSpectrogram;
pub use types::{DelayPreset, StreamingConfig, StreamingStats, TranscriptionEvent};
