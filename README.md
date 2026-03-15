# 🧬 Phi-4 Vision + Audio mmproj Generator

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-compatible-green.svg)](https://github.com/ggml-org/llama.cpp)
[![Model: Phi-4](https://img.shields.io/badge/Model-Phi--4%20Multimodal-purple.svg)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

**Extract vision and audio projectors from Microsoft's Phi-4 multimodal models into llama.cpp-compatible GGUF files.**

*Supports both `Phi4MMForCausalLM` (Phi-4-multimodal-instruct) and `Phi4ForCausalLMV` (Phi-4-reasoning-vision).*

</div>

---

## ✨ What This Does

Microsoft's Phi-4 family includes two multimodal architectures:

- **[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)** (`Phi4MMForCausalLM`) — 5.6B model with vision **and** audio (SigLIP-2 + Conformer)
- **[Phi-4-reasoning-vision](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)** (`Phi4ForCausalLMV`) — 15B reasoning model with vision (SigLIP-2 NaFlex)

Both models can be converted to llama.cpp's GGUF format for local inference, but the built-in `convert_hf_to_gguf.py` does not produce mmproj files for them. This script fills that gap.

It extracts the multimodal encoders and projectors into a standalone **mmproj GGUF file**, which you pass to `llama-server` or `llama-cli` alongside the quantized text model.

### Three Output Modes

| Mode | Encoders Included | Use Case |
|:---:|---|---|
| 👁️ **Vision** | SigLIP-2 ViT + MLP projector | Image understanding, VQA, chart reading |
| 🎤 **Audio** | Conformer (24 layers) + MLP projector | ASR, translation, summarization |
| 🌐 **Omni** | Both vision + audio in a single GGUF | Full Phi-4 multimodal capabilities |

> **Note:** Audio mode is only available for `Phi4MMForCausalLM`. `Phi4ForCausalLMV` is vision-only.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install safetensors numpy

# Recommended: PyTorch for bfloat16 support
# Phi-4-reasoning-vision stores weights in bfloat16 — numpy alone cannot handle this.
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> If PyTorch is not installed, the script falls back to numpy. This works for Phi-4-multimodal-instruct (float16/float32 weights) but will fail on Phi-4-reasoning-vision (bfloat16 weights).

### 2. Run the Converter

```bash
# Interactive mode (recommended for first-time use)
python convert_phi4_mmproj.py --model-dir /path/to/Phi-4-multimodal-instruct

# Phi-4-reasoning-vision (auto-detected, vision only)
python convert_phi4_mmproj.py --model-dir /path/to/Phi-4-reasoning-vision-15B --mode vision

# Direct mode — audio only (Phi4MM only)
python convert_phi4_mmproj.py --model-dir ./merged --mode audio

# Direct mode — omni (vision + audio combined, Phi4MM only)
python convert_phi4_mmproj.py --model-dir ./merged --mode omni -o phi4-omni.gguf
```

When run without `--mode`, the script shows an **interactive selection menu** that auto-detects available modalities:

```
╔══════════════════════════════════════════════════════════════╗
║  Phi-4 Multimodal mmproj — Mode Selection                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [1] 👁️  Vision only   — SigLIP-2 encoder + MLP projector  ║
║      Image understanding, VQA, chart reading                ║
║                                                              ║
║  [2] 🎤 Audio only    — Conformer encoder + MLP projector  ║
║      Speech recognition (ASR), translation, summarization  ║
║                                                              ║
║  [3] 🌐 Omni (both)   — Vision + Audio in one GGUF         ║
║      Full multimodal: images AND speech                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

> If `--mode omni` is requested but the model has no audio tensors (e.g. Phi-4-reasoning-vision), the script automatically falls back to `--mode vision`.

### 3. Use with llama.cpp

```bash
# Server mode (OpenAI-compatible API)
llama-server \
  -m phi4-text-Q4_K_M.gguf \
  --mmproj mmproj-phi4-vision-f16.gguf \
  --host 0.0.0.0 --port 8080

# CLI mode
llama-cli \
  -m phi4-text-Q4_K_M.gguf \
  --mmproj mmproj-phi4-vision-f16.gguf \
  --prompt "Describe this image:"
```

---

## 🏗️ Architecture

### Phi4MMForCausalLM — Vision Encoder (SigLIP-2 with HD tiling)

| Component | Details |
|---|---|
| Architecture | Vision Transformer (ViT) |
| Encoder layers | 27 |
| Hidden dim | 1152 |
| Attention heads | 16 |
| Image size | 448 × 448 px |
| Patch size | 14 × 14 px |
| Projection | 2-layer MLP → 3072 (LLM dim) |
| HD feature | Global + sub-image separators (`glb_GN`, `sub_GN`) |

### Phi4ForCausalLMV — Vision Encoder (SigLIP-2 NaFlex, variable resolution)

| Component | Details |
|---|---|
| Architecture | Vision Transformer (ViT) |
| Vision tower | `google/siglip2-so400m-patch16-naflex` |
| Encoder layers | 27 (last layer dropped per llama.cpp PR #20168) |
| Hidden dim | 1152 |
| Attention heads | 16 |
| Image size | Variable (NaFlex) — nominal 448 px |
| Patch size | 16 × 16 px |
| Projection | 2-layer MLP GELU → 5120 (LLM dim), `projector_type="phi4"` |
| Tensor dtype | bfloat16 → requires PyTorch for conversion |

### Phi4MMForCausalLM — Audio Encoder (Conformer)

| Component | Details |
|---|---|
| Architecture | Conformer (Transformer + Conv) |
| Encoder layers | 24 |
| Hidden dim | 512 |
| Attention heads | 8 |
| Input | 80-bin mel spectrogram |
| FFN style | Macaron (dual FFN: in + out) |
| Convolution | DepthWise Separable Conv1D + GLU gating |
| Attention bias | T5-style relative attention bias |
| Embedding | 3-layer Conv1D front-end |
| Projection | 2-layer MLP → 3072 (speech + vision cross-modal) |

---

## 🗺️ Tensor Mapping

### Phi4MMForCausalLM — Vision Tensors
```
model.embed_tokens_extend.image_embed.img_processor.encoder.layers.{N}.self_attn.q_proj.weight
  → v.blk.{N}.attn_q.weight

model.embed_tokens_extend.image_embed.img_projection.0.weight
  → mm.0.weight
```

### Phi4ForCausalLMV — Vision Tensors
```
# Note the double vision_tower nesting:
#   model.vision_tower          = SiglipVisionTower wrapper (nn.Module)
#   .vision_tower               = actual Siglip2VisionModel (self.vision_tower attribute)
#   .vision_model.encoder.*     = SigLIP-2 encoder weights

model.vision_tower.vision_tower.vision_model.encoder.layers.{N}.self_attn.q_proj.weight
  → v.blk.{N}.attn_q.weight

model.mm_projector.0.weight
  → mm.0.weight
```

### Phi4MMForCausalLM — Audio Tensors
```
model.embed_tokens_extend.audio_embed.encoder.encoders.{N}.self_attn.linear_q.weight
  → a.blk.{N}.attn_q.weight

model.embed_tokens_extend.audio_embed.encoder.relative_attention_bias_layer.bias_values.weight
  → a.rel_attn_bias

model.embed_tokens_extend.audio_embed.audio_projection.speech.0.weight
  → mm.a.mlp.0.weight
```

> Tensor mappings were derived by cross-referencing the [Phi-4 safetensors index](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), [vLLM phi4mm.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi4mm.py), [llama.cpp tensor_mapping.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/tensor_mapping.py), [llama.cpp PR #20168](https://github.com/ggml-org/llama.cpp/pull/20168) (Phi4ForCausalLMV support), and the [ShayanCyan GGUF reference](https://github.com/Ahmed-Shayan-Arsalan/Phi4-multimodal-Quantisized-Llama.cpp).

---

## ⚙️ Options

```
usage: convert_phi4_mmproj.py [-h] --model-dir MODEL_DIR
                               [--mode {vision,audio,omni}]
                               [--output OUTPUT]
                               [--dtype {f16,f32}]
                               [--verbose]

options:
  --model-dir MODEL_DIR   Path to merged model directory (required)
  --mode {vision,audio,omni}
                          Output mode (default: interactive menu)
  --output, -o OUTPUT     Output GGUF path (default: auto-named)
  --dtype {f16,f32}       Tensor precision (default: f16)
  --verbose, -v           Print full tensor mapping
```

---

## 🔌 llama.cpp Compatibility

| Feature | Status | Notes |
|---|:---:|---|
| 👁️ Vision inference (Phi4MM) | ✅ **Works** | Mainline llama.cpp |
| 👁️ Vision inference (Phi4V) | ✅ **Works** | Requires [llama.cpp PR #20168](https://github.com/ggml-org/llama.cpp/pull/20168) (merged Mar 2026) |
| 🎤 Audio inference | ⚠️ **Fork required** | Needs C++ Conformer runtime |
| 🌐 Omni GGUF generation | ✅ **Works** | This script generates it |

### Audio: Required Fork

Audio inference requires a custom llama.cpp build with Conformer C++ support:

👉 **[Ahmed-Shayan-Arsalan/Phi4-multimodal-Quantisized-Llama.cpp](https://github.com/Ahmed-Shayan-Arsalan/Phi4-multimodal-Quantisized-Llama.cpp)**

> **Tip:** Generate the omni GGUF now — vision works immediately, and audio will activate automatically once Conformer support lands in mainline llama.cpp (active community PRs in progress).

---

## 🧩 Script Design

This script is designed to be **self-contained** and **PR-quality**:

- **No `gguf-py` dependency** — includes a minimal GGUF writer from scratch (standard library + numpy only)
- **Only `safetensors` + `numpy` required** — PyTorch optional but recommended for bfloat16
- **Auto-detects architecture** — reads `config.json` to distinguish `Phi4MMForCausalLM` vs `Phi4ForCausalLMV`
- **Auto-detects inner prefix nesting** — handles both the standard `vision_model.*` layout and the double-nested `vision_tower.vision_model.*` layout used by Phi4ForCausalLMV
- **Conv weights kept in F32** — for numerical stability during quantization
- **Post-generation instructions** — prints mode-specific llama.cpp usage examples after conversion

The design follows the patterns of `convert_hf_to_gguf.py` to be easily incorporated into the official llama.cpp repository as a native `Phi4MMForCausalLM` / `Phi4ForCausalLMV` mmproj exporter.

---

## 📋 Supported Models

| HuggingFace Architecture | Model | Text GGUF | mmproj |
|---|---|:---:|:---:|
| `Phi4MMForCausalLM` | [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | ✅ | ✅ Vision + Audio |
| `Phi4ForCausalLMV` | [Phi-4-reasoning-vision-15B](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B) | ✅ | ✅ Vision |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Model weights are subject to [Microsoft's Phi-4 License](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/LICENSE).

---

## 🙏 Acknowledgements

- **Microsoft** — for the Phi-4 multimodal model family
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — the GGUF runtime
- **[Ahmed-Shayan-Arsalan](https://github.com/Ahmed-Shayan-Arsalan/Phi4-multimodal-Quantisized-Llama.cpp)** — reference implementation for the Conformer audio encoder mapping
- **[vLLM team](https://github.com/vllm-project/vllm)** — phi4mm.py tensor naming reference
