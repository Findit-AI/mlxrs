# Changelog

## [0.2.0] — 2026-08-31

The models arrive. 0.1.0 shipped the support surface and deliberately no
architectures; 0.2.0 adds eleven of them, every one opt-in behind a feature —
Whisper, Qwen3-ASR, SenseVoice, wav2vec2, Silero VAD, CLAP, EmbeddingGemma,
SigLIP2 NaFlex, Qwen3, LFM2, and LFM2.5-VL — on top of new core work
(convolutions, `mx.compile`, a shared quantized-layer module, a config/weight
validation toolkit, and NumPy `.npy` / `.npz` I/O).

### Added

#### Core

- **`ops::conv`** — `conv1d` / `conv2d` / `conv3d`, the three `conv_transpose*` counterparts, and the fully parameterized `conv_general`.
- **`transforms::compile`** — `mx.compile` graph compilation: `compile` / `compile_fn` build a reusable `Compiled` closure over `&[Array] -> Vec<Array>`, with a process-wide `CompileMode` and `enable_compile` / `disable_compile` / `set_compile_mode`.
- **`nn`** (new top-level module) — dense + quantized layers shared by every model family: `Linear`, `QuantizedLinear`, `QuantizedEmbedding`, and the load-time-polymorphic `MaybeQuantizedLinear` / `MaybeQuantizedEmbedding` that let one code path load a dense, 8-bit, or 4-bit checkpoint. Deliberately outside `lm::nn` so `embeddings` (which does not enable `lm`) can reach it.
- **`model_validation`** (new top-level module) — the shared config/weight validation toolkit every ported model loads through: `pin_*` exact-value pins, `require_positive` / `require_in_range` / `require_divisible` / `require_cardinality` range guards, overflow-checked `checked_mul` / `checked_add`, and the resource-safety trio `Extent` / `elem_count` / `alloc_filled` plus `reserve_or_error` (the `TryReserve` trait) so a hostile checkpoint header cannot drive an unbounded allocation.
- **`io` NumPy support** — `load_npy` / `load_npz` / `save_npy` / `save_npz` / `save_npz_compressed` behind the new `npz` feature, mirroring `mx.save` / `savez` / `savez_compressed` (stored + DEFLATE members); loads are memory-mapped, cutting peak load memory from ~3x to ~1x.
- **`io::load_weights_from_dir`** — one feature-neutral, format-detecting loader over a checkpoint directory: the `model.safetensors.index.json` shard map, plain safetensors, GGUF, and `.npz`. `lm::load` now delegates to it instead of carrying its own discovery tier.

#### Language models

- **LFM2** — the hybrid short-conv + attention LM (`lfm2-vl`), including the `model_type` resolution and RoPE-null fallback the reference checkpoints need.
- **Qwen3** — the dense text transformer (`qwen3`): GQA with per-head Q/K RMSNorm, SwiGLU MLP, tied output head.
- **Quantized checkpoints across the stack** — Qwen3, Whisper, wav2vec2, SigLIP2, and Qwen3-ASR all load through the shared `MaybeQuantizedLinear` / `MaybeQuantizedEmbedding` path, with a `.scales`-presence discriminator that decides dense-vs-quantized from the checkpoint itself.

#### Vision-language models

- **LFM2.5-VL** — the vision-language wrapper over LFM2 (`lfm2-vl`), with tiling config and native-resolution image handling.
- **`vlm::model::ImageProcessor`** — a per-model preprocessing seam, so a native-resolution processor is a model's own concern rather than a branch in the shared path.

#### Speech-to-text

- **The golden STT trait architecture** — `Transcribe` + `TranscribeExt`, specialized by `CtcModel` and `AutoregressiveStt`, with `ForcedAligner<Input>` for timestamp alignment; the shared vocabulary is `TranscribeOptions`, `Transcription`, `Segment`, `Task`, and `AlignOptions` / `AlignedSpan` / `ForcedAlignment`. Every shipped STT model implements it, so callers swap architectures without touching call sites.
- **Whisper** (`whisper`) — OpenAI Whisper on the golden traits, with word-level timestamps via cross-attention DTW (`WordTiming`) and the accompanying hallucination heuristics; `condition_on_previous_text`, `initial_prompt`, and `clip_timestamps` for mlx-audio parity; batched best-of-N decoding through `MaximumLikelihoodRanker`; AlignAtt streaming (`WhisperStreaming`, `StreamingOptions`, `CommittedToken`, `StreamingStep`); and fp16/bf16 support via activation-dtype preservation.
- **A CoreML / Neural-Engine Whisper backend** — drives compiled WhisperKit `.mlmodelc` MelSpectrogram / AudioEncoder / TextDecoder models on the ANE through pure `objc2-core-ml`. Selected at load time from the checkpoint shape via `WhisperBackend`, platform-gated to `aarch64-apple-darwin` rather than exposed as a cargo feature.
- **wav2vec2** (`wav2vec2`) — CTC speech-to-text, generic over a `Family` dialect trait that covers base / large plus the HuBERT variants: per-layer `feat_extract_norm`, MMS adapters, and HuBERT's no-LayerNorm projection.
- **Qwen3-ASR** (`qwen3-asr`) — the Conv2d-stem audio encoder + transformer, spliced into the Qwen3 text decoder; the forced aligner behind `qwen3-asr-aligner` runs a timestamp-classification head over the decoder and owns word splitting through the model tokenizer.
- **SenseVoice** (`sensevoice`) — SenseVoice-Small CTC: Kaldi-fbank + LFR + CMVN front end, SANM/FSMN encoder, CTC head, and the rich-info language / emotion / event predictions.

#### Voice activity detection

- **Silero VAD** (`vad`) — 1:1 mlx-audio parity: per-rate STFT-conv stem, four conv blocks, LSTM, sigmoid speech-probability head, fixed 512/256-sample chunking at 16k/8k with carried streaming context, and threshold / min-speech / min-silence / speech-pad segment post-processing. Public surface is `VadModel`, `VadOutput`, and `SpeechSegment`.

#### Embeddings

- **SigLIP2 NaFlex** (`siglip2-naflex`) — the dual-tower embedder on the golden `Embed` / `Contrastive` traits, with NaFlex variable-resolution preprocessing reusing the `vlm::resize` NEON path.
- **EmbeddingGemma** (`embeddinggemma`) — the Gemma3 bidirectional backbone with mean pooling and the Dense projection head, including the bidirectional sliding-window mask on local layers.
- **CLAP** (`clap`) — CLAP-HTSAT-unfused audio+text embeddings: Slaney mel front end over `audio::dsp`, HTSAT Swin audio tower, RoBERTa text tower, and zero-shot classification. Numerically pinned against committed `.npy` oracles.

#### SIMD

- **`rgba_to_rgb_affine`** — a NEON patchify-normalize kernel (with the usual scalar fallback) for the SigLIP2 preprocessing path, plus a `simd_siglip_normalize` benchmark.

#### Build & deps

- **docs.rs builds again** — `mlxrs-sys`'s build script now returns early under `DOCS_RS` (rustdoc never links the native library, and the pre-committed bindings suffice), and both manifests point docs.rs at `aarch64-apple-darwin`.
- **`serde_json` is a real feature**, not a bare `dep:serde_json` — so `#[cfg(feature = "serde_json")]` is an accurate gate, and `io::load_weights_from_dir` can gate its JSON index tier without forcing JSON onto the `serde_json`-free `embeddings` build.
- New optional dependencies: `flate2` (Whisper's decode compression-ratio heuristic), `memmapix` (memory-mapped `.npy` / `.npz`), `unicode-general-category` (the aligner's exact General_Category keep-predicate), `zip` 8 (`npz` members), and the `objc2` / `objc2-foundation` / `objc2-core-ml` trio behind the Apple-silicon target gate.

### Changed

- **BREAKING — `cpal` 0.17 → 0.18.** `cpal` types appear in the public playback surface, so the bump is a caller-visible break: `AudioPlayer::with_device` now takes a 0.18 `&cpal::Device`, and `PlaybackConfig::cpal_config` returns a 0.18 `cpal::StreamConfig`.
- **BREAKING — the async playback error changed type.** The typed device error boxed into `Error::ExternalOp`'s source chain is now `cpal::Error` (was `cpal::StreamError`). The documented recovery — `payload.inner().downcast_ref::<StreamError>()` — no longer matches; downcast to `cpal::Error` and read `kind()` / `message()` instead.
- **BREAKING — `StandardKvCache` is a step buffer**, matching mlx-lm's `KVCache` and mlx-swift. Its growth and trim semantics differ from 0.1.0's for callers driving a cache directly.
- **`mlxrs` requires `mlxrs-sys` 0.2** — the sys crate is republished for the docs.rs build fix.
- **`zip` 2 → 8** for the `npz` feature.

### Fixed

- **TTS G2P** — inline `#` comments are stripped from CMUDict pronunciations, so a canonical `cmudict.dict` loads. This is the one entry here a 0.1.0 user can actually hit; every correction below landed on code this same cycle introduced, and is recorded because the reference-parity work is the interesting part of these ports, not because 0.1.0 shipped the bug.
- **Dtype discipline at every model entry** — the mel is cast to the model dtype in Whisper's `encode`, the log-mel at CLAP's audio-tower entry, spliced audio rows to the embedding dtype in Qwen3-ASR, and `pixel_values` to the model dtype in SigLIP2 NaFlex. Each was a silent fp32/fp16 mismatch on a non-fp32 checkpoint.
- **Whisper** — a double off-by-one in the `max_decoder_ctx` prefill guard and the `sample_len` cap (a regression from the CoreML backend work).
- **LFM2-VL** — flagship-checkpoint loading and the reference dtype casts restored; `eos_token_id` is now null-tolerant and resolved through `text_config`.
- **SigLIP2 NaFlex** — pad with Gemma's `<pad>` = 0, not SigLIP1's EOS = 1.
- **CLAP** — text-tower pooling, `layer_norm_eps`, and the L2-normalize epsilon corrected against the reference.
- **Qwen3-ASR** — `sanitize`'s conv transpose is gated on `is_formatted`.
- **Qwen3 / ForcedAligner** — load-time soundness guards, and the aligner's knobs moved onto a typed Options type.
- **Silero VAD** — the audit findings from the port review.
- **`io::npy`** — handle `zip` 8.6.0's `data_start()` returning `Option<u64>`.

### Performance

- **Whisper decode is several times faster** — one GPU sync per greedy token (previously several), a pipelined decode loop built on lazy primitives + `async_eval` + on-device state, fused SDPA for encoder self-attention, and logit filters moved onto the GPU.
- **SenseVoice** — rich-info argmax batched from three GPU syncs to one, and LFR vectorized.
- **CLAP** — the Swin SW-MSA shifted-window mask is built once at construction instead of per forward.
- **LFM2-VL** — fused-addmm `Linear`, an all-zeros vision-mask skip, and a bicubic identity fast path.
- **EmbeddingGemma** — activations and `clip_residual` fused through `mx.compile`.

## [0.1.0] — 2026-05-31

First release. Safe Rust bindings for Apple's MLX array framework on Apple
silicon (`aarch64-apple-darwin`) via the `mlx-c` FFI layer, plus opt-in
higher-level support surfaces ported from MLX's companion projects.

### Added

#### Core

- **`mlxrs-sys`** — pre-committed bindgen output for `mlx-c`. Builds vendored mlx-c (+ gguflib) via cmake-rs; links libmlxc + libmlx + Metal/Accelerate.
- **`Array`** RAII handle, the `Dtype` enum, and the `Element` trait (bool, integer, and float / half / bfloat / complex element types).
- **Lazy evaluation** (`Array::eval`) — reading data via `item` / `to_vec` forces evaluation. `Array` does **not** implement `Clone` (the only duplication is the fallible `try_clone`) and is `!Send + !Sync` (single-thread, like MLX's own APIs); move results across threads as owned data from `to_vec` / `item`.
- **Ops** across `arithmetic`, `reduction`, `comparison`, `logical`, `shape`, `indexing`, `linalg`, `fft`, `fast`, and `misc`, plus `transforms` (autodiff + graph transformations) and a `memory` API.
- **Public `Stream` / `Device` API** — thread-affine, non-RAII handles for explicit stream/device placement.
- **`io`** — safetensors and GGUF load/save.
- **`simd`** — arch-gated NEON kernels (with scalar fallbacks) for hot image/audio paths.
- **Operator overloads** (`&a + &b`, `-&a`, …) gated behind the off-by-default `unstable-ops-overload` feature; they **panic** on shape/dtype error, so library authors must never enable them transitively — the fallible `a.add(&b)?` form is the load-bearing API.

#### Optional feature surfaces (off by default)

- **`lm`** — language models: HF tokenizers (BPE / SentencePiece / chat templates / tool-call parsing), KV-caches (rotating / chunked / batched / quantized), samplers + logits processors, quantization, LoRA / DoRA, optimizers (Adam, AdamW, Adamax, Adafactor, Muon, SGD, Lion, Adadelta, RMSprop), and the generation loop + chat session.
- **`vlm`** — vision-language models (implies `lm`): image preprocessing, prompt assembly, multimodal generation.
- **`audio`** — audio (implies `lm`): STFT / mel DSP, WAV I/O, STT / TTS serializers, playback.
- **`embeddings`** — embedding-model loading, pooling modes, and the encode pipeline.
- **`llguidance`** — grammar-constrained / structured decoding.
- **`gguf`** — GGUF load/save (gguflib is vendored + statically linked by `mlxrs-sys`).
- Finer-grained `tokenizer-*` flags expose individual tokenizer pieces without the full `lm` surface.

#### Tooling & CI

- **xtask `regen-bindings`** — re-run bindgen against the vendored mlx-c headers.
- Per-crate CI (`mlxrs-sys.yml`, `mlxrs.yml`) with matrix feature builds, clippy / fmt / docs gates, a bindings-drift gate, a coverage job, and a weekly `dep-watch.yml` dependency cron.
- Extensive unit-test coverage (~90% of the testable surface; device-bound playback / Metal-kernel dispatch and unreachable defensive guards are excluded).

### Architecture decisions

- **`aarch64-apple-darwin` only.** Other targets (`x86_64-apple-darwin`, Linux + CUDA, distributed) are roadmapped.
- **No dependency pinning** in any manifest (loose semver only); `Cargo.lock` is gitignored. The reproducibility check is the weekly `dep-watch.yml` cron, not a checked-in lockfile.
- **Pre-committed bindings + a drift CI gate** anchor binding stability — `regen-bindings` must be re-run + committed when the mlx-c submodule moves.
- **Async Metal kernel failures intentionally abort the process.** The rc/sentinel chain only catches synchronous errors; a `set_terminate`-style recovery shim is not implementable (mlx-c exposes no hook), and only diagnostics are planned.
- **No per-model architectures are bundled** — the `lm` / `vlm` / `audio` / `embeddings` features ship the support surface (loaders, tokenizers, caches, samplers, processors, generation loops, audio I/O), not specific model implementations; those are added per use-case.

### Safety audits

- Entry refcount audit (`docs/audits/send-soundness.md`, local-only) — verified MLX `array_desc_` is a `std::shared_ptr` with an atomic refcount, but per-clone `Send` is unsound because `set_status` mutates non-atomic state through `const`. Final design is `!Send + !Sync`.
- Adversarial code review on every PR — caught and fixed empty-slice dangling-pointer UB across `slice` / `sum_axes` / `concatenate` / `gather` / `pad` / shape-taking ops; introduced `dim_ptr` / `data_ptr` / per-`Element` `sentinel_ptr` helpers; sealed `IntoShape`; centralized `validate_dims` at every FFI boundary; routed empty-axes reductions through MLX; and added bounded overflow / cap guards on FFI op wrappers that reach unchecked C++ arithmetic.

[0.2.0]: https://github.com/findit-studio/mlxrs/releases/tag/v0.2.0
[0.1.0]: https://github.com/findit-studio/mlxrs/releases/tag/v0.1.0
