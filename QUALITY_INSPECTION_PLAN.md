# mlxrs 封装质量质检计划 v2 — 细粒度版

**目标：** 逐模块对照上游源码，用 4 专家团并行审查，验证正确性、封装质量、性能和 Rust 实践。
**时间：** 7 天（2026-05-28 ~ 2026-06-03）
**方法：** 每个模块独立审查，4 个专家团并行出报告，合并发现。

---

## 专家团设计（每模块 4 团并行）

### 团 1: Faithfulness（忠实度专家）
**身份：** 你是一个资深的 Python/Swift → Rust 移植审查员。你的职责是逐行对比上游源码和 mlxrs 实现，确保移植忠实。
**检查项：**
- 上游每个函数/方法是否都有对应的 Rust 实现？
- 分支逻辑是否完整？（if/else/elif 是否全部覆盖）
- 数值计算是否一致？（常量、公式、舍入）
- 上游的注释/文档中提到的 edge case 是否处理了？
- 上游的 bug 是否照搬了？上游的 fix 是否同步了？
- 上游的默认值是否一致？

### 团 2: Rust Quality（Rust 质量专家）
**身份：** 你是一个有 10 年经验的 Rust 工程师，专注于 API 设计和惯用法。你的职责是审查 Rust 代码是否利用了语言优势。
**检查项：**
- 所有权设计是否合理？（有无不必要的 Arc/Mutex/Rc）
- 生命周期是否清晰？（有无不必要的 'static 或过度约束）
- 错误处理是否一致？（Result + ? vs unwrap/expect）
- trait 设计是否合理？（sealed vs open、object-safe、blanket impl）
- 命名是否符合 Rust 规范？（snake_case、TypeSuffix、SCREAMING_SNAKE_CASE）
- 是否有不必要的 pub 暴露？
- 文档注释是否完整？（/// 和 //! 用法）

### 团 3: Performance（性能专家）
**身份：** 你是一个专注于 Apple Silicon 性能优化的工程师。你的职责是找出性能瓶颈和优化机会。
**检查项：**
- 有无不必要的 .clone() 或 .to_vec()？
- 有无不必要的 heap 分配？（能用 slice 的地方用了 Vec？）
- 热路径上是否有 Box<dyn Trait> 导致的动态分派？
- SIMD 利用率如何？有无遗漏的 SIMD 优化机会？
- 是否有可以用 rayon 并行化的串行循环？
- 对比 Python/Swift：哪些地方应该更快？哪些地方反而更慢？
- 有无可以用 const generics 替代 runtime 参数的地方？

### 团 4: Adversarial（对抗专家）
**身份：** 你是一个专门找 bug 的代码审查员。你的职责是主动寻找错误、反模式和潜在问题。
**检查项：**
- 有无照搬 Python 的可变全局状态模式？（应该用 Rust 的类型系统）
- 有无照搬 Python 的 duck typing 而忽略了类型安全？
- 有无 panic 路径在生产代码中？（unwrap/expect/index）
- 有无死代码/未完成的功能？
- 有无逻辑错误？（条件反转、off-by-one、遗漏的 break）
- 有无安全问题？（unsafe 使用、整数溢射、DoS 向量）
- 有无上游 API 变更后未同步的风险？

---

## 模块清单（25 个审查单元）

### Day 1: 核心层（3 rounds）

**QCR-1: array/ + ops/ 基础**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| array/ (construction, conversion, arithmetic, linalg) | 1,882 | 9 | mlx-c |
| ops/ (arithmetic, comparison, reduction, shape, random, linalg, fft) | 5,131 | 13 | mlx-c + mlx-lm |

**QCR-2: ops/fast + transforms + memory**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| ops/fast/ (metal_kernel, quantized) | 1,008 | 2 | mlx-lm quantized.py |
| transforms/ (autograd, checkpoint, closure) | 1,925 | 6 | mlx-c transforms |
| memory/ (wired, counters) | 1,103 | 4 | mlx-c memory |

**QCR-3: 底层工具（error, dtype, device, stream, shape, io）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| error.rs | 116,655 | 1 | mlxrs-original |
| dtype.rs | 19,728 | 1 | mlx-c |
| device.rs | 15,174 | 1 | mlx-c |
| stream.rs | 30,425 | 1 | mlx-c |
| shape.rs | 5,675 | 1 | mlx-c |
| io.rs | 62,035 | 1 | mlx-lm gguf.py + safetensors |

### Day 2: LM 核心（3 rounds）

**QCR-4: lm/cache/（KV 缓存 — 最复杂的模块之一）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| cache/standard.rs | ~2,500 | 1 | mlx-lm cache.py (KVCache) |
| cache/quantized.rs | ~2,500 | 1 | mlx-lm cache.py (QuantizedKVCache) |
| cache/batch_rotating.rs | 1,384 | 1 | mlx-lm cache.py (BatchedKVCache) |
| cache/lru_cache.rs | ~1,500 | 1 | mlx-swift-lm KVCache.swift |
| cache/mod.rs + traits | ~2,800 | 9 | mlx-lm + mlx-swift-lm |

**QCR-5: lm/load.rs + lm/gguf.rs（模型加载）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| lm/load.rs | 5,908 | 1 | mlx-lm utils.py (load, _get_classes) |
| lm/gguf.rs | 2,300 | 1 | mlx-lm gguf.py |
| lm/convert.rs | 3,976 | 1 | mlx-lm convert.py |

**QCR-6: lm/generate.rs + lm/session.rs（生成 + 会话）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| lm/generate.rs | 4,017 | 1 | mlx-lm generate.py |
| lm/session.rs | 2,783 | 1 | mlx-lm cache_prompt.py + generate.py |

### Day 3: LM 训练 + 模型（3 rounds）

**QCR-7: lm/lora.rs + lm/quant.rs（微调 + 量化）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| lm/lora.rs | 8,635 | 1 | mlx-lm lora.py + dora.py |
| lm/quant.rs | 5,093 | 1 | mlx-lm quantized.py + fuse.py |

**QCR-8: lm/tuner/（训练系统）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| tuner/datasets.rs | 2,022 | 1 | mlx-lm datasets.py |
| tuner/trainer.rs | 1,932 | 1 | mlx-lm trainer.py |
| tuner/losses.rs | 1,594 | 1 | mlx-lm losses.py |
| tuner/dora.rs + optimizers.rs | ~60 | 2 | mlx-lm dora.py, optimizers.py |

**QCR-9: lm/nn/（神经网络层）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| nn/switch.rs | 2,622 | 1 | mlx-lm switch_layers.py |
| nn/norm.rs | 1,636 | 1 | mlx-lm normalization.py |
| nn/rope_scaling.rs | 1,615 | 1 | mlx-lm rope_utils.py |
| nn/attention.rs, rope.rs, linear.rs | ~1,700 | 4 | mlx-lm base.py |

### Day 4: LM 模型架构 + VLM（3 rounds）

**QCR-10: lm/model.rs + lm/factory.rs（模型定义 + 工厂）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| lm/model.rs | ~2,000 | 1 | mlx-lm models/base.py |
| lm/factory.rs | ~3,000 | 1 | mlx-lm utils.py + mlx-swift-lm |
| lm/tool_parsers.rs | 64 | 1 | mlx-lm tool_parsers/ |

**QCR-11: vlm/ 核心（load, generate, prompt, inputs）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| vlm/load.rs | 3,375 | 1 | mlx-vlm utils.py |
| vlm/generate.rs | 1,015 | 1 | mlx-vlm generate.py |
| vlm/prompt.rs | 2,053 | 1 | mlx-vlm utils.py (prompt building) |
| vlm/inputs.rs | 635 | 1 | mlx-vlm utils.py |

**QCR-12: vlm/ 媒体处理（image, video, resize, feature_cache）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| vlm/image.rs | 3,008 | 1 | mlx-vlm utils.py (image processing) |
| vlm/resize.rs | 1,491 | 1 | mlx-vlm resize logic |
| vlm/video.rs | 946 | 1 | mlx-vlm video processing |
| vlm/feature_cache.rs | 1,074 | 1 | mlx-vlm vision_cache.py |

### Day 5: 音频（3 rounds）

**QCR-13: audio/dsp.rs + audio/features.rs（DSP 核心）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| audio/dsp.rs | 6,315 | 1 | mlx-audio dsp.py |
| audio/features.rs | 3,227 | 1 | mlx-audio dsp.py (Kaldi features) |

**QCR-14: audio/io.rs + audio/load.rs + audio/playback/（IO + 播放）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| audio/io.rs | 2,958 | 1 | mlx-audio audio_io.py |
| audio/load.rs | 796 | 1 | mlx-audio utils.py |
| audio/playback/ | 1,701 | 4 | mlx-audio-swift AudioPlayer |

**QCR-15: audio/stt/ + audio/tts/ + audio/sts/ + audio/vad/ + audio/lid/ + audio/codec/**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| audio/stt/ | 2,908 | 5 | mlx-audio stt/ + mlx-lm whisper.py |
| audio/tts/ | 1,788 | 5 | mlx-audio tts/ + mlx-lm kokoro.py |
| audio/sts/ | 244 | 2 | mlx-lm voice_pipeline.py |
| audio/vad/ | 368 | 3 | mlx-lm silero_vad.py |
| audio/lid/ | 383 | 3 | mlx-lm vec_lid.py |
| audio/codec/ | 226 | 2 | mlx-audio codec/ |

### Day 6: Tokenizer + Embeddings + SIMD（3 rounds）

**QCR-16: tokenizer/（分词器）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| tokenizer/ (HfTokenizer, SentencePiece, streaming detok, GPT-2, chat template) | 14,032 | 8 | mlx-lm tokenizer + mlx-audio-swift SentencePiece |

**QCR-17: embeddings/（嵌入模型）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| embeddings/factory.rs | 3,409 | 1 | mlx-swift-lm Embedders |
| embeddings/config.rs | 1,640 | 1 | mlx-swift-lm ModelConfiguration |
| embeddings/colvision.rs | 1,282 | 1 | mlx-swift-lm ColVision |
| embeddings/encode.rs | 954 | 1 | mlx-swift-lm EmbedderModelContainer |
| embeddings/pooling.rs, model.rs, similarity.rs, etc. | 1,360 | 6 | mlx-swift-lm |

**QCR-18: simd/（SIMD 引擎 — mlxrs-original）**
| 模块 | 行数 | 文件 | 上游 |
|------|------|------|------|
| simd/audio/ (mel, kaldi_mel, pcm_decode, resample, quantize, window, lfilter) | 4,876 | 8 | mlx-audio dsp.py (纯 Rust 重写) |
| simd/vlm/ (bgr_widen, rgb_widen, rotate_buf, pad_canvas_fill) | 2,685 | 4 | mlx-vlm image processing (纯 Rust 重写) |
| simd/dispatch + arch + scalar | 1,352 | 6 | mlxrs-original |
| simd/diff.rs | ~200 | 1 | mlxrs-original |

### Day 7: 交叉审查 + 汇总（2 rounds）

**QCR-19: 跨模块一致性审查**
- 错误处理模式是否全局一致？
- FFI 调用模式是否全局一致？（RAII → check() → stream）
- 命名风格是否全局一致？
- 文档风格是否全局一致？
- Feature gate 设计是否合理？

**QCR-20: 汇总 + 报告**
- 合并所有 QCR 发现
- 按严重度排序
- 生成优化建议清单
- 提交 GitHub Issue

---

## 每日执行节奏

```
Day 1: QCR-1 ~ QCR-3   (核心层)           3 rounds × 4 teams = 12 份报告
Day 2: QCR-4 ~ QCR-6   (LM cache/load/gen) 3 rounds × 4 teams = 12 份报告
Day 3: QCR-7 ~ QCR-9   (LM 训练/模型)     3 rounds × 4 teams = 12 份报告
Day 4: QCR-10 ~ QCR-12 (LM 架构 + VLM)    3 rounds × 4 teams = 12 份报告
Day 5: QCR-13 ~ QCR-15 (音频)             3 rounds × 4 teams = 12 份报告
Day 6: QCR-16 ~ QCR-18 (Tokenizer+Emb+SIMD) 3 rounds × 4 teams = 12 份报告
Day 7: QCR-19 ~ QCR-20 (交叉审查+汇总)    2 rounds
```

**总计：20 个 QCR × 4 个专家团 = 80 份独立审查报告**

---

## 执行约束

- 每个 QCR 开始前，先 `git pull origin main` 确保代码最新
- 每个专家团读文件时，先读上游再读 mlxrs（顺序重要）
- 发现问题必须给出具体的代码位置（文件:行号）和修复建议
- 优化建议必须说明 Rust 优势在哪里（零成本抽象/所有权/SIMD/etc.）
- 不要泛泛而谈，每个发现都要有具体的代码引用
