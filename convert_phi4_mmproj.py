#!/usr/bin/env python3
"""
convert_phi4_mmproj.py — Extract multimodal projectors from Phi-4 models.

Converts the SigLIP-2 vision encoder, Conformer audio encoder, and their
projection MLPs from Phi-4 safetensors into mmproj GGUF files compatible
with llama.cpp.

Supports three output modes:
  • vision  — Vision encoder + image projection (SigLIP-2)
  • audio   — Audio encoder + speech projection (Conformer)
  • omni    — Both vision + audio in a single GGUF

Supported architectures:
  - Phi4MMForCausalLM  (Phi-4-multimodal-instruct)
  - Phi4ForCausalLMV   (Phi-4-reasoning-vision-15B)

Usage:
  # Interactive mode selection:
  python convert_phi4_mmproj.py --model-dir /path/to/merged

  # Direct mode:
  python convert_phi4_mmproj.py --model-dir /path/to/merged --mode omni

  # Vision only to specific output:
  python convert_phi4_mmproj.py --model-dir /path/to/merged --mode vision -o mmproj-vision.gguf

Author: Christian (UltimateTradingBot project)
License: MIT
"""

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)

# PyTorch is used for bfloat16 support — numpy cannot handle bfloat16 natively.
# The Phi-4-reasoning-vision model stores tensors in bfloat16.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

def _st_open(path):
    """Open safetensors file. Use torch framework if available (handles bfloat16)."""
    if _TORCH_AVAILABLE:
        return safe_open(path, framework="pt")
    return safe_open(path, framework="numpy")

def _st_to_numpy(tensor, output_dtype: np.dtype) -> np.ndarray:
    """Convert safetensors tensor to numpy array, handling bfloat16."""
    if _TORCH_AVAILABLE:
        # torch tensor — convert bfloat16 to float16/float32 before numpy()
        if tensor.dtype == _torch.bfloat16:
            tensor = tensor.to(_torch.float32)
        arr = tensor.numpy()
    else:
        arr = tensor  # already numpy
    if arr.dtype != output_dtype:
        return arr.astype(output_dtype)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# GGUF Format Constants
# Reference: ggml-org/llama.cpp/gguf-py/gguf/constants.py
# ─────────────────────────────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

class GGUFValueType:
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

class GGMLType:
    F32  = 0
    F16  = 1
    BF16 = 30


# ─────────────────────────────────────────────────────────────────────────────
# Phi-4 → GGUF Tensor Name Mappings
# Derived from cross-referencing:
#   1. microsoft/Phi-4-multimodal-instruct/model.safetensors.index.json
#   2. vllm/model_executor/models/phi4mm.py (hf_to_vllm_mapper)
#   3. ggml-org/llama.cpp/gguf-py/gguf/tensor_mapping.py
#   4. Ahmed-Shayan-Arsalan/Phi4-multimodal-Quantisized-Llama.cpp (conformer)
# ─────────────────────────────────────────────────────────────────────────────

PHI4_VISION_PREFIX = "model.embed_tokens_extend.image_embed."
PHI4_AUDIO_PREFIX  = "model.embed_tokens_extend.audio_embed."

# ── Phi4ForCausalLMV (Phi-4-reasoning-vision) ──
# Uses LLaVA-style architecture: separate vision_tower + mm_projector
PHI4V_TOWER_PREFIX      = "model.vision_tower."
PHI4V_PROJECTOR_PREFIX  = "model.mm_projector."

# Encoder layer map (relative path after stripping tower prefix + inner sub-prefix)
# Same GGUF names as Phi4MM — only source path differs
PHI4V_ENCODER_LAYER_MAP = {
    "self_attn.q_proj.weight":   "v.blk.{bid}.attn_q.weight",
    "self_attn.q_proj.bias":     "v.blk.{bid}.attn_q.bias",
    "self_attn.k_proj.weight":   "v.blk.{bid}.attn_k.weight",
    "self_attn.k_proj.bias":     "v.blk.{bid}.attn_k.bias",
    "self_attn.v_proj.weight":   "v.blk.{bid}.attn_v.weight",
    "self_attn.v_proj.bias":     "v.blk.{bid}.attn_v.bias",
    "self_attn.out_proj.weight": "v.blk.{bid}.attn_out.weight",
    "self_attn.out_proj.bias":   "v.blk.{bid}.attn_out.bias",
    "layer_norm1.weight":        "v.blk.{bid}.ln1.weight",
    "layer_norm1.bias":          "v.blk.{bid}.ln1.bias",
    "layer_norm2.weight":        "v.blk.{bid}.ln2.weight",
    "layer_norm2.bias":          "v.blk.{bid}.ln2.bias",
    "mlp.fc1.weight":            "v.blk.{bid}.ffn_up.weight",
    "mlp.fc1.bias":              "v.blk.{bid}.ffn_up.bias",
    "mlp.fc2.weight":            "v.blk.{bid}.ffn_down.weight",
    "mlp.fc2.bias":              "v.blk.{bid}.ffn_down.bias",
}

PHI4V_EMBEDDING_MAP = {
    "embeddings.patch_embedding.weight":      "v.patch_embd.weight",
    "embeddings.patch_embedding.bias":        "v.patch_embd.bias",
    "embeddings.position_embedding.weight":   "v.position_embd.weight",
}

PHI4V_POST_NORM_MAP = {
    "post_layernorm.weight": "v.post_ln.weight",
    "post_layernorm.bias":   "v.post_ln.bias",
}

PHI4V_PROJECTOR_MAP = {
    "0.weight": "mm.0.weight",
    "0.bias":   "mm.0.bias",
    "2.weight": "mm.2.weight",
    "2.bias":   "mm.2.bias",
}

# ── Vision: SigLIP-2 encoder (27 layers) ──

VISION_ENCODER_LAYER_MAP = {
    "self_attn.q_proj.weight":   "v.blk.{bid}.attn_q.weight",
    "self_attn.q_proj.bias":     "v.blk.{bid}.attn_q.bias",
    "self_attn.k_proj.weight":   "v.blk.{bid}.attn_k.weight",
    "self_attn.k_proj.bias":     "v.blk.{bid}.attn_k.bias",
    "self_attn.v_proj.weight":   "v.blk.{bid}.attn_v.weight",
    "self_attn.v_proj.bias":     "v.blk.{bid}.attn_v.bias",
    "self_attn.out_proj.weight": "v.blk.{bid}.attn_out.weight",
    "self_attn.out_proj.bias":   "v.blk.{bid}.attn_out.bias",
    "layer_norm1.weight":        "v.blk.{bid}.ln1.weight",
    "layer_norm1.bias":          "v.blk.{bid}.ln1.bias",
    "layer_norm2.weight":        "v.blk.{bid}.ln2.weight",
    "layer_norm2.bias":          "v.blk.{bid}.ln2.bias",
    "mlp.fc1.weight":            "v.blk.{bid}.ffn_up.weight",
    "mlp.fc1.bias":              "v.blk.{bid}.ffn_up.bias",
    "mlp.fc2.weight":            "v.blk.{bid}.ffn_down.weight",
    "mlp.fc2.bias":              "v.blk.{bid}.ffn_down.bias",
}

VISION_EMBEDDING_MAP = {
    "img_processor.embeddings.patch_embedding.weight": "v.patch_embd.weight",
    "img_processor.embeddings.patch_embedding.bias":   "v.patch_embd.bias",
    "img_processor.embeddings.position_embedding.weight": "v.position_embd.weight",
}

VISION_POST_NORM_MAP = {
    "img_processor.post_layernorm.weight": "v.post_ln.weight",
    "img_processor.post_layernorm.bias":   "v.post_ln.bias",
}

VISION_PROJECTOR_MAP = {
    "img_projection.0.weight": "mm.0.weight",
    "img_projection.0.bias":   "mm.0.bias",
    "img_projection.2.weight": "mm.2.weight",
    "img_projection.2.bias":   "mm.2.bias",
}

VISION_HD_MAP = {
    "glb_GN": "v.glb_GN",
    "sub_GN": "v.sub_GN",
}

# ── Audio: Conformer encoder (24 layers) ──
# Phi-4 uses a custom Conformer (WeNet/Cascades-style) with:
#   - Macaron FFN (feed_forward_in + feed_forward_out)
#   - DepthWiseSeparableConv1d (dw_conv + pw_conv)
#   - GLU gating (glu.b1, glu.b2, glu.ext_pw_conv_1d)
#   - T5-style relative attention bias
#   - Convolutional embedding (conv layers)

AUDIO_CONV_EMBED_MAP = {
    # Convolutional embedding front-end (3 conv layers → linear out)
    "encoder.embed.conv.0.weight":  "a.conv1d.0.weight",
    "encoder.embed.conv.0.bias":    "a.conv1d.0.bias",
    "encoder.embed.conv.2.weight":  "a.conv1d.2.weight",
    "encoder.embed.conv.2.bias":    "a.conv1d.2.bias",
    "encoder.embed.conv.3.weight":  "a.conv1d.3.weight",  # BatchNorm weight
    "encoder.embed.conv.3.bias":    "a.conv1d.3.bias",    # BatchNorm bias
    "encoder.embed.conv.5.weight":  "a.conv1d.5.weight",
    "encoder.embed.conv.5.bias":    "a.conv1d.5.bias",
    "encoder.embed.conv.6.weight":  "a.conv1d.6.weight",  # BatchNorm weight
    "encoder.embed.conv.6.bias":    "a.conv1d.6.bias",    # BatchNorm bias
    "encoder.embed.out.weight":     "a.conv1d.out.weight",
    "encoder.embed.out.bias":       "a.conv1d.out.bias",
    # Global CMVN normalization
    "encoder.encoder_embedding.global_mean":   "a.global_mean",
    "encoder.encoder_embedding.global_invstd": "a.global_invstd",
}

AUDIO_ENCODER_LAYER_MAP = {
    # Self-attention
    "self_attn.linear_q.weight":   "a.blk.{bid}.attn_q.weight",
    "self_attn.linear_q.bias":     "a.blk.{bid}.attn_q.bias",
    "self_attn.linear_k.weight":   "a.blk.{bid}.attn_k.weight",
    "self_attn.linear_k.bias":     "a.blk.{bid}.attn_k.bias",
    "self_attn.linear_v.weight":   "a.blk.{bid}.attn_v.weight",
    "self_attn.linear_v.bias":     "a.blk.{bid}.attn_v.bias",
    "self_attn.linear_out.weight": "a.blk.{bid}.attn_out.weight",
    "self_attn.linear_out.bias":   "a.blk.{bid}.attn_out.bias",
    # Layer norms
    "layer_norm.weight":           "a.blk.{bid}.ln.weight",
    "layer_norm.bias":             "a.blk.{bid}.ln.bias",
    "layer_norm_att.weight":       "a.blk.{bid}.ln_att.weight",
    "layer_norm_att.bias":         "a.blk.{bid}.ln_att.bias",
    # Macaron FFN — feed_forward_in (first half)
    "feed_forward_in.layer_norm.weight":       "a.blk.{bid}.ffn_in.ln.weight",
    "feed_forward_in.layer_norm.bias":         "a.blk.{bid}.ffn_in.ln.bias",
    "feed_forward_in.net.0.linear.weight":     "a.blk.{bid}.ffn_in.up.weight",
    "feed_forward_in.net.0.linear.bias":       "a.blk.{bid}.ffn_in.up.bias",
    "feed_forward_in.net.2.weight":            "a.blk.{bid}.ffn_in.down.weight",
    "feed_forward_in.net.2.bias":              "a.blk.{bid}.ffn_in.down.bias",
    # Macaron FFN — feed_forward_out (second half)
    "feed_forward_out.layer_norm.weight":      "a.blk.{bid}.ffn_out.ln.weight",
    "feed_forward_out.layer_norm.bias":        "a.blk.{bid}.ffn_out.ln.bias",
    "feed_forward_out.net.0.linear.weight":    "a.blk.{bid}.ffn_out.up.weight",
    "feed_forward_out.net.0.linear.bias":      "a.blk.{bid}.ffn_out.up.bias",
    "feed_forward_out.net.2.weight":           "a.blk.{bid}.ffn_out.down.weight",
    "feed_forward_out.net.2.bias":             "a.blk.{bid}.ffn_out.down.bias",
    # Convolution module — depthwise separable conv
    "conv.layer_norm.weight":                         "a.blk.{bid}.conv.ln.weight",
    "conv.layer_norm.bias":                           "a.blk.{bid}.conv.ln.bias",
    "conv.ext_pw_conv_1d.weight":                     "a.blk.{bid}.conv.pw_ext.weight",
    "conv.ext_pw_conv_1d.bias":                       "a.blk.{bid}.conv.pw_ext.bias",
    "conv.glu.ext_pw_conv_1d.weight":                 "a.blk.{bid}.conv.glu.pw.weight",
    "conv.glu.ext_pw_conv_1d.bias":                   "a.blk.{bid}.conv.glu.pw.bias",
    "conv.glu.b1":                                    "a.blk.{bid}.conv.glu.b1",
    "conv.glu.b2":                                    "a.blk.{bid}.conv.glu.b2",
    "conv.dw_sep_conv_1d.dw_conv.weight":             "a.blk.{bid}.conv.dw.weight",
    "conv.dw_sep_conv_1d.dw_conv.bias":               "a.blk.{bid}.conv.dw.bias",
    "conv.dw_sep_conv_1d.pw_conv.weight":             "a.blk.{bid}.conv.pw_mid.weight",
    "conv.dw_sep_conv_1d.pw_conv.bias":               "a.blk.{bid}.conv.pw_mid.bias",
}

AUDIO_REL_ATTN_MAP = {
    "encoder.relative_attention_bias_layer.bias_values.weight": "a.rel_attn_bias",
}

# Audio projection MLP (2-layer: Linear → GELU → Linear)
# Has separate projections for speech→text and vision→audio cross-modal
AUDIO_PROJECTOR_MAP = {
    "audio_projection.speech.0.weight": "mm.a.mlp.0.weight",
    "audio_projection.speech.0.bias":   "mm.a.mlp.0.bias",
    "audio_projection.speech.2.weight": "mm.a.mlp.2.weight",
    "audio_projection.speech.2.bias":   "mm.a.mlp.2.bias",
    # Vision-audio cross-modal projection
    "audio_projection.vision.0.weight": "mm.a.vis.0.weight",
    "audio_projection.vision.0.bias":   "mm.a.vis.0.bias",
    "audio_projection.vision.2.weight": "mm.a.vis.2.weight",
    "audio_projection.vision.2.bias":   "mm.a.vis.2.bias",
}


# ─────────────────────────────────────────────────────────────────────────────
# Vision / Audio Config Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VISION_CONFIG = {
    "image_size": 448,
    "patch_size": 14,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "layer_norm_eps": 1e-6,
    "projection_dim": 3072,
}

SIGLIP_IMAGE_MEAN = [0.5, 0.5, 0.5]
SIGLIP_IMAGE_STD  = [0.5, 0.5, 0.5]

DEFAULT_AUDIO_CONFIG = {
    "hidden_size": 512,            # Conformer d_model
    "num_hidden_layers": 24,       # Conformer encoder layers
    "num_attention_heads": 8,      # Conformer attention heads
    "intermediate_size": 2048,     # Conformer FFN dim
    "layer_norm_eps": 1e-5,
    "num_mel_bins": 80,            # Mel spectrogram bins
    "projection_dim": 3072,        # Output projection to LLM hidden
}


# ─────────────────────────────────────────────────────────────────────────────
# GGUF Writer (self-contained, no gguf-py dependency)
# ─────────────────────────────────────────────────────────────────────────────

class GGUFWriter:
    def __init__(self, path: str):
        self.path = path
        self.kv_data: list[tuple[str, int, Any]] = []
        self.tensors: list[tuple[str, np.ndarray]] = []

    def add_string(self, key: str, value: str):
        self.kv_data.append((key, GGUFValueType.STRING, value))

    def add_uint32(self, key: str, value: int):
        self.kv_data.append((key, GGUFValueType.UINT32, value))

    def add_int32(self, key: str, value: int):
        self.kv_data.append((key, GGUFValueType.INT32, value))

    def add_float32(self, key: str, value: float):
        self.kv_data.append((key, GGUFValueType.FLOAT32, value))

    def add_bool(self, key: str, value: bool):
        self.kv_data.append((key, GGUFValueType.BOOL, value))

    def add_array(self, key: str, values: list, elem_type: int):
        self.kv_data.append((key, GGUFValueType.ARRAY, (elem_type, values)))

    def add_tensor(self, name: str, data: np.ndarray):
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        self.tensors.append((name, data))

    def _encode_string(self, s: str) -> bytes:
        encoded = s.encode('utf-8')
        return struct.pack('<Q', len(encoded)) + encoded

    def _encode_kv_value(self, vtype: int, value: Any) -> bytes:
        if vtype == GGUFValueType.STRING:
            return self._encode_string(value)
        elif vtype == GGUFValueType.UINT32:
            return struct.pack('<I', value)
        elif vtype == GGUFValueType.INT32:
            return struct.pack('<i', value)
        elif vtype == GGUFValueType.FLOAT32:
            return struct.pack('<f', value)
        elif vtype == GGUFValueType.BOOL:
            return struct.pack('<B', 1 if value else 0)
        elif vtype == GGUFValueType.UINT64:
            return struct.pack('<Q', value)
        elif vtype == GGUFValueType.ARRAY:
            elem_type, values = value
            buf = struct.pack('<II', elem_type, len(values))
            for v in values:
                buf += self._encode_kv_value(elem_type, v)
            return buf
        else:
            raise ValueError(f"Unsupported GGUF value type: {vtype}")

    def _get_ggml_type(self, dtype: np.dtype) -> int:
        if dtype == np.float32:
            return GGMLType.F32
        elif dtype == np.float16:
            return GGMLType.F16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def write(self):
        ALIGNMENT = 32
        with open(self.path, 'wb') as f:
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.kv_data)))

            for key, vtype, value in self.kv_data:
                f.write(self._encode_string(key))
                f.write(struct.pack('<I', vtype))
                f.write(self._encode_kv_value(vtype, value))

            tensor_infos = []
            offset = 0
            for name, data in self.tensors:
                ggml_type = self._get_ggml_type(data.dtype)
                n_dims = len(data.shape)
                tensor_infos.append((name, data, ggml_type, n_dims, offset))
                data_size = data.nbytes
                offset += data_size
                padding = (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT
                offset += padding

            for name, data, ggml_type, n_dims, tensor_offset in tensor_infos:
                f.write(self._encode_string(name))
                f.write(struct.pack('<I', n_dims))
                for dim in reversed(data.shape):
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', ggml_type))
                f.write(struct.pack('<Q', tensor_offset))

            current_pos = f.tell()
            padding = (ALIGNMENT - (current_pos % ALIGNMENT)) % ALIGNMENT
            f.write(b'\x00' * padding)

            for name, data, ggml_type, n_dims, tensor_offset in tensor_infos:
                f.write(data.tobytes())
                data_size = data.nbytes
                padding = (ALIGNMENT - (data_size % ALIGNMENT)) % ALIGNMENT
                f.write(b'\x00' * padding)


# ─────────────────────────────────────────────────────────────────────────────
# Config Extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_configs(model_dir: Path, arch: str = "phi4mm") -> tuple[dict, dict]:
    """Extract vision and audio configs from Phi-4 config.json."""
    config_path = model_dir / "config.json"
    vision_cfg = DEFAULT_VISION_CONFIG.copy()
    audio_cfg = DEFAULT_AUDIO_CONFIG.copy()

    if not config_path.exists():
        print(f"  ⚠️  No config.json found, using defaults")
        return vision_cfg, audio_cfg

    with open(config_path) as f:
        cfg = json.load(f)

    embd = cfg.get("embd_layer", {})

    # Vision — Phi4MM path
    img_embd = embd.get("image_embd_layer", {})
    if crop_size := img_embd.get("crop_size"):
        vision_cfg["image_size"] = crop_size

    # Phi4ForCausalLMV: vision_config sub-dict
    # Config from Phi-4-reasoning-vision (siglip2_vision_model) contains:
    #   hidden_size, intermediate_size, num_attention_heads, num_hidden_layers
    # but NOT image_size or patch_size (NaFlex = variable resolution).
    vision_config = cfg.get("vision_config", {})
    if not img_embd and vision_config:
        if img_size := vision_config.get("image_size"):
            vision_cfg["image_size"] = img_size
        if patch_size := vision_config.get("patch_size"):
            vision_cfg["patch_size"] = patch_size
        if num_layers := vision_config.get("num_hidden_layers"):
            vision_cfg["num_hidden_layers"] = num_layers
        if h_size := vision_config.get("hidden_size"):
            vision_cfg["hidden_size"] = h_size
        if int_size := vision_config.get("intermediate_size"):
            vision_cfg["intermediate_size"] = int_size
        if num_heads := vision_config.get("num_attention_heads"):
            vision_cfg["num_attention_heads"] = num_heads

    # Extract patch_size from mm_vision_tower name if not in vision_config
    # e.g. "google/siglip2-so400m-patch16-naflex" → patch_size=16
    if vision_cfg["patch_size"] == DEFAULT_VISION_CONFIG["patch_size"]:
        mm_tower = cfg.get("mm_vision_tower", "")
        import re as _re
        m_ps = _re.search(r"patch(\d+)", mm_tower)
        if m_ps:
            vision_cfg["patch_size"] = int(m_ps.group(1))

    # The projection output dim matches the LLM hidden_size
    if hidden_size := cfg.get("hidden_size"):
        vision_cfg["projection_dim"] = hidden_size
        audio_cfg["projection_dim"] = hidden_size

    return vision_cfg, audio_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Tensor Extraction
# ─────────────────────────────────────────────────────────────────────────────

def find_safetensor_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        for subdir in ["merged", "output/merged"]:
            sub = model_dir / subdir
            if sub.exists():
                files = sorted(sub.glob("*.safetensors"))
                if files:
                    break
    return files


def _map_vision_tensor(rel_name: str) -> str | None:
    """Map a vision tensor relative name to GGUF name."""
    # Encoder layers
    m = re.match(r"img_processor\.encoder\.layers\.(\d+)\.(.+)", rel_name)
    if m:
        layer_idx, suffix = int(m.group(1)), m.group(2)
        if suffix in VISION_ENCODER_LAYER_MAP:
            return VISION_ENCODER_LAYER_MAP[suffix].format(bid=layer_idx)

    if rel_name in VISION_EMBEDDING_MAP:
        return VISION_EMBEDDING_MAP[rel_name]
    if rel_name in VISION_POST_NORM_MAP:
        return VISION_POST_NORM_MAP[rel_name]
    if rel_name in VISION_PROJECTOR_MAP:
        return VISION_PROJECTOR_MAP[rel_name]
    if rel_name in VISION_HD_MAP:
        return VISION_HD_MAP[rel_name]
    return None


def _map_audio_tensor(rel_name: str) -> str | None:
    """Map an audio tensor relative name to GGUF name."""
    # Encoder layers
    m = re.match(r"encoder\.encoders\.(\d+)\.(.+)", rel_name)
    if m:
        layer_idx, suffix = int(m.group(1)), m.group(2)
        if suffix in AUDIO_ENCODER_LAYER_MAP:
            return AUDIO_ENCODER_LAYER_MAP[suffix].format(bid=layer_idx)

    if rel_name in AUDIO_CONV_EMBED_MAP:
        return AUDIO_CONV_EMBED_MAP[rel_name]
    if rel_name in AUDIO_REL_ATTN_MAP:
        return AUDIO_REL_ATTN_MAP[rel_name]
    if rel_name in AUDIO_PROJECTOR_MAP:
        return AUDIO_PROJECTOR_MAP[rel_name]
    return None


def _detect_phi4v_inner_prefix(safetensor_files: list) -> str:
    """Detect the sub-prefix within model.vision_tower.

    Phi4ForCausalLMV (Phi-4-reasoning-vision) nests the SigLIP model as:
      model.vision_tower        (= Siglip2VisionTower wrapper class)
        .vision_tower           (= actual Siglip2VisionModel stored as self.vision_tower)
          .vision_model.*       (= Siglip2VisionModel state dict keys)

    So actual keys: model.vision_tower.vision_tower.vision_model.encoder.layers.0.*
    Inner prefix:   "vision_tower.vision_model."

    Older/alternate checkpoints may use "vision_model." directly (no double nesting).
    Returns the detected inner prefix string.
    """
    for st_path in safetensor_files:
        with _st_open(st_path) as st:
            for key in st.keys():
                if key.startswith(PHI4V_TOWER_PREFIX):
                    rel = key[len(PHI4V_TOWER_PREFIX):]
                    # Double-nested: model.vision_tower.vision_tower.vision_model.*
                    if rel.startswith("vision_tower.vision_model.") or rel.startswith("vision_tower.encoder."):
                        return "vision_tower.vision_model."
                    # Single-nested: model.vision_tower.vision_model.*
                    if rel.startswith("vision_model."):
                        return "vision_model."
                    # No wrapper: model.vision_tower.encoder.* / embeddings.* / post_layernorm.*
                    if rel.startswith("encoder.") or rel.startswith("embeddings.") or rel.startswith("post_layernorm"):
                        return ""
    # Default: Phi4ForCausalLMV uses double-nesting via Siglip2VisionTower
    return "vision_tower.vision_model."


def _map_phi4v_tensor(rel: str, is_projector: bool = False) -> str | None:
    """Map a Phi4ForCausalLMV tensor to GGUF name.
    rel: path relative to tower prefix + inner sub-prefix (for vision)
         OR path relative to PHI4V_PROJECTOR_PREFIX (for projector)
    """
    if is_projector:
        return PHI4V_PROJECTOR_MAP.get(rel)

    # Encoder block layers
    m = re.match(r"encoder\.layers\.(\d+)\.(.+)", rel)
    if m:
        layer_idx, suffix = int(m.group(1)), m.group(2)
        if suffix in PHI4V_ENCODER_LAYER_MAP:
            return PHI4V_ENCODER_LAYER_MAP[suffix].format(bid=layer_idx)

    if rel in PHI4V_EMBEDDING_MAP:
        return PHI4V_EMBEDDING_MAP[rel]
    if rel in PHI4V_POST_NORM_MAP:
        return PHI4V_POST_NORM_MAP[rel]
    return None


def extract_tensors(
    safetensor_files: list[Path],
    include_vision: bool,
    include_audio: bool,
    output_dtype: np.dtype = np.float16,
    arch: str = "phi4mm",
    inner_prefix: str = "vision_model.",
    vision_cfg: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    """Extract and rename modality tensors from Phi-4 safetensors."""
    tensors: dict[str, np.ndarray] = {}
    mapping_log: dict[str, str] = {}
    skipped: list[str] = []

    _phi4v_full_tower = PHI4V_TOWER_PREFIX + inner_prefix
    _vcfg = vision_cfg or {}
    # Determine total encoder layers to know which to skip (last layer dropped per PR #20168)
    _phi4v_num_layers = _vcfg.get("num_hidden_layers", 27) if arch == "phi4v" else 0
    _phi4v_patch_size = _vcfg.get("patch_size", 16) if arch == "phi4v" else 16

    for st_path in safetensor_files:
        print(f"  📂 Scanning {st_path.name}...")
        with _st_open(st_path) as st:
            for key in st.keys():
                gguf_name = None

                if arch == "phi4v":
                    # Phi4ForCausalLMV: vision tower + mm_projector
                    if include_vision and key.startswith(_phi4v_full_tower):
                        rel = key[len(_phi4v_full_tower):]
                        # Skip last encoder layer — Phi-4 uses hidden_states[-2] (PR #20168)
                        m_skip = re.match(r"encoder\.layers\.(\d+)\.", rel)
                        if m_skip and int(m_skip.group(1)) >= _phi4v_num_layers - 1:
                            skipped.append(key)
                            continue
                        # Skip post_layernorm and head (not used in Phi-4 mmproj)
                        if "post_layernorm" in rel or ".head." in rel:
                            skipped.append(key)
                            continue
                        gguf_name = _map_phi4v_tensor(rel, is_projector=False)
                        if gguf_name is None:
                            skipped.append(key)
                            continue
                    elif include_vision and key.startswith(PHI4V_PROJECTOR_PREFIX):
                        rel = key[len(PHI4V_PROJECTOR_PREFIX):]
                        gguf_name = _map_phi4v_tensor(rel, is_projector=True)
                        if gguf_name is None:
                            skipped.append(key)
                            continue
                    else:
                        continue
                else:
                    # Phi4MMForCausalLM: embed_tokens_extend architecture
                    if include_vision and key.startswith(PHI4_VISION_PREFIX):
                        rel = key[len(PHI4_VISION_PREFIX):]
                        gguf_name = _map_vision_tensor(rel)
                        if gguf_name is None:
                            skipped.append(key)
                            continue
                    elif include_audio and key.startswith(PHI4_AUDIO_PREFIX):
                        rel = key[len(PHI4_AUDIO_PREFIX):]
                        gguf_name = _map_audio_tensor(rel)
                        if gguf_name is None:
                            skipped.append(key)
                            continue
                    else:
                        continue

                # _st_to_numpy handles bfloat16 (torch) and dtype conversion
                conv_dtype = np.float32 if "conv" in gguf_name.lower() else output_dtype
                tensor = _st_to_numpy(st.get_tensor(key), conv_dtype)

                # Phi4V patch embedding: reshape [D, C*P²] → [D, C, P, P] (PR #20168)
                if arch == "phi4v" and gguf_name == "v.patch_embd.weight" and tensor.ndim == 2:
                    d, cpq = tensor.shape
                    p = _phi4v_patch_size
                    c = cpq // (p * p)
                    tensor = tensor.reshape(d, p, p, c).transpose(0, 3, 1, 2)  # [D,C,P,P]

                tensors[gguf_name] = tensor
                mapping_log[key] = gguf_name

    if skipped:
        print(f"\n  ⚠️  Skipped {len(skipped)} unmapped tensors:")
        for s in skipped[:8]:
            print(f"      {s}")
        if len(skipped) > 8:
            print(f"      ... and {len(skipped) - 8} more")

    # Diagnostic: if phi4v and 0 vision encoder tensors extracted, show actual key structure
    if arch == "phi4v" and include_vision:
        v_count = sum(1 for n in tensors if n.startswith("v.blk."))
        if v_count == 0:
            print(f"\n  ⚠️  DIAGNOSTIC: 0 vision encoder tensors extracted for phi4v arch!")
            print(f"      Expected prefix: '{_phi4v_full_tower}'")
            print(f"      Scanning safetensors for actual keys under model.vision_tower.*...")
            samples = []
            for st_path in safetensor_files[:2]:
                with _st_open(st_path) as st:
                    for key in st.keys():
                        if "vision_tower" in key or "vision_model" in key:
                            samples.append(key)
                            if len(samples) >= 15:
                                break
                if len(samples) >= 15:
                    break
            if samples:
                print(f"      First {len(samples)} matching keys:")
                for s in samples:
                    print(f"        {s}")
            else:
                print(f"      No keys containing 'vision_tower' or 'vision_model' found!")

    return tensors, mapping_log, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Menu
# ─────────────────────────────────────────────────────────────────────────────

def show_menu(has_vision: bool, has_audio: bool) -> str:
    """Show interactive mode selection menu. Returns mode string."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phi-4 Multimodal mmproj — Mode Selection                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║                                                              ║")
    if has_vision:
        print("║  [1] 👁️  Vision only   — SigLIP-2 encoder + MLP projector  ║")
        print("║      Image understanding, VQA, chart reading                ║")
    else:
        print("║  [1] 👁️  Vision only   — ⚠️  No vision tensors found       ║")
    print("║                                                              ║")
    if has_audio:
        print("║  [2] 🎤 Audio only    — Conformer encoder + MLP projector  ║")
        print("║      Speech recognition (ASR), translation, summarization  ║")
    else:
        print("║  [2] 🎤 Audio only    — ⚠️  No audio tensors found         ║")
    print("║                                                              ║")
    if has_vision and has_audio:
        print("║  [3] 🌐 Omni (both)   — Vision + Audio in one GGUF         ║")
        print("║      Full multimodal: images AND speech                     ║")
    else:
        print("║  [3] 🌐 Omni (both)   — ⚠️  Requires both modalities       ║")
    print("║                                                              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  ℹ️  Compatibility                                           ║")
    print("║                                                              ║")
    print("║  👁️  Vision works with mainline llama.cpp (llama-server).    ║")
    print("║                                                              ║")
    print("║  🎤 Audio requires C++ Conformer support.                   ║")
    print("║     Use this fork with audio support:                       ║")
    print("║     https://github.com/Ahmed-Shayan-Arsalan/               ║")
    print("║       Phi4-multimodal-Quantisized-Llama.cpp                 ║")
    print("║     Or wait for mainline llama.cpp to merge Conformer PRs.  ║")
    print("║                                                              ║")
    print("║  🌐 Omni GGUF: vision works immediately, audio once         ║")
    print("║     Conformer support lands in your llama.cpp build.        ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            choice = input("  Select mode [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            sys.exit(0)

        if choice == "1":
            if not has_vision:
                print("  ❌ No vision tensors found in this model.")
                continue
            return "vision"
        elif choice == "2":
            if not has_audio:
                print("  ❌ No audio tensors found in this model.")
                continue
            return "audio"
        elif choice == "3":
            if not (has_vision and has_audio):
                print("  ❌ Need both vision and audio tensors for omni mode.")
                continue
            return "omni"
        else:
            print("  ❌ Invalid selection. Enter 1, 2, or 3.")


def detect_modalities(safetensor_files: list[Path]) -> tuple[bool, bool, str]:
    """Quick check for presence of vision/audio tensors.
    Returns (has_vision, has_audio, arch) where arch is 'phi4mm', 'phi4v', or 'unknown'.
    """
    has_vision = False
    has_audio = False
    arch = "unknown"
    prefix_samples: set[str] = set()

    for st_path in safetensor_files:
        with _st_open(st_path) as st:
            for key in st.keys():
                # Collect first 4 path components for diagnostics
                parts = key.split(".")
                prefix_samples.add(".".join(parts[:min(4, len(parts))]))

                if key.startswith(PHI4_VISION_PREFIX):
                    has_vision = True
                    arch = "phi4mm"
                elif key.startswith(PHI4_AUDIO_PREFIX):
                    has_audio = True
                    if arch == "unknown":
                        arch = "phi4mm"
                elif key.startswith(PHI4V_TOWER_PREFIX) or key.startswith(PHI4V_PROJECTOR_PREFIX):
                    has_vision = True
                    arch = "phi4v"

                if has_vision and has_audio:
                    return True, True, arch

    if not has_vision and not has_audio:
        print()
        print("  ⚠️  Known prefixes not found. Tensor prefixes actually present:")
        top_prefixes: dict[str, int] = {}
        for p in prefix_samples:
            top = ".".join(p.split(".")[:2])
            top_prefixes[top] = top_prefixes.get(top, 0) + 1
        for p, count in sorted(top_prefixes.items(), key=lambda x: -x[1])[:20]:
            print(f"      {p}  ({count} tensors)")
        print(f"  Expected vision prefix: '{PHI4_VISION_PREFIX}' (Phi4MM)")
        print(f"  Expected vision prefix: '{PHI4V_TOWER_PREFIX}' (Phi4V)")
        print(f"  Expected audio prefix:  '{PHI4_AUDIO_PREFIX}'")

    return has_vision, has_audio, arch


# ─────────────────────────────────────────────────────────────────────────────
# Usage Instructions
# ─────────────────────────────────────────────────────────────────────────────

def print_usage_instructions(mode: str, output_path: str, file_size_mb: float, n_tensors: int):
    """Print post-generation instructions for llama.cpp usage."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ✅ mmproj GGUF created successfully!                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  File:     {output_path}")
    print(f"  Size:     {file_size_mb:.1f} MB")
    print(f"  Tensors:  {n_tensors}")
    print(f"  Mode:     {mode}")
    print()

    mmproj = os.path.basename(output_path)

    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  How to use with llama.cpp                                  │")
    print("├──────────────────────────────────────────────────────────────┤")
    print("│                                                              │")
    print("│  1. Server mode (recommended):                              │")
    print("│                                                              │")
    print(f"│     llama-server \\                                          │")
    print(f"│       -m text-model.gguf \\                                  │")
    print(f"│       --mmproj {mmproj} \\                          │")
    print(f"│       --host 0.0.0.0 --port 8080                           │")
    print("│                                                              │")
    print("│  2. CLI mode:                                               │")
    print("│                                                              │")
    print(f"│     llama-cli \\                                             │")
    print(f"│       -m text-model.gguf \\                                  │")
    print(f"│       --mmproj {mmproj}                            │")
    print("│                                                              │")

    if mode == "vision":
        print("│  3. Test vision:                                            │")
        print("│     Send an image in the chat or via API:                   │")
        print("│     curl http://localhost:8080/v1/chat/completions \\        │")
        print("│       -d '{ \"messages\": [{ \"role\": \"user\",               │")
        print("│              \"content\": [                                   │")
        print("│                { \"type\": \"image_url\",                      │")
        print("│                  \"image_url\": { \"url\": \"file://img.jpg\" }},│")
        print("│                { \"type\": \"text\",                           │")
        print("│                  \"text\": \"What do you see?\" }             │")
        print("│              ] }] }'                                        │")

    elif mode == "audio":
        print("│  3. Test audio (experimental):                              │")
        print("│     Audio support requires llama.cpp with libmtmd.          │")
        print("│     Send WAV/MP3 audio in chat prompts.                     │")
        print("│                                                              │")
        print("│  ⚠️  NOTE: Audio inference requires C++ conformer support.  │")
        print("│     Use the fork by Ahmed-Shayan-Arsalan or wait for        │")
        print("│     mainline llama.cpp to merge conformer support.          │")

    elif mode == "omni":
        print("│  3. Capabilities:                                           │")
        print("│     • Images:  VQA, captioning, chart/doc understanding     │")
        print("│     • Audio:   ASR, translation, speech summarization       │")
        print("│     Note: Use text+images OR text+audio per prompt,         │")
        print("│     not all three simultaneously.                           │")
        print("│                                                              │")
        print("│  ⚠️  NOTE: Audio features require C++ conformer support.   │")
        print("│     Vision works with mainline llama.cpp.                   │")
        print("│     Audio needs Ahmed-Shayan-Arsalan's fork or future PR.  │")

    print("│                                                              │")
    print("└──────────────────────────────────────────────────────────────┘")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract mmproj (vision/audio/omni projector) GGUF from Phi-4 multimodal models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (shows menu):
  python convert_phi4_mmproj.py --model-dir /workspace/output/merged

  # Vision only:
  python convert_phi4_mmproj.py --model-dir ./merged --mode vision

  # Audio only:
  python convert_phi4_mmproj.py --model-dir ./merged --mode audio

  # Omni (vision + audio combined):
  python convert_phi4_mmproj.py --model-dir ./merged --mode omni -o phi4-omni.gguf

  # Pipeline mode (non-interactive, defaults to omni):
  python convert_phi4_mmproj.py --model-dir ./merged --mode omni --dtype f16
""",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to the model directory containing safetensor files",
    )
    parser.add_argument(
        "--mode", "-m", type=str, choices=["vision", "audio", "omni"],
        default=None,
        help="Output mode: vision, audio, or omni (default: interactive)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output GGUF file path (default: auto-named based on mode)",
    )
    parser.add_argument(
        "--dtype", type=str, choices=["f16", "f32"], default="f16",
        help="Output tensor dtype (default: f16)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print tensor mapping details",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"❌ Error: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dtype = np.float32 if args.dtype == "f32" else np.float16

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phi-4 Multimodal → mmproj GGUF Converter                  ║")
    print("║  Supports: Vision (SigLIP-2) + Audio (Conformer)           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Model dir:  {model_dir}")
    print(f"  Dtype:      {args.dtype}")
    print()

    # 1. Find safetensor files
    st_files = find_safetensor_files(model_dir)
    if not st_files:
        print(f"❌ No safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(st_files)} safetensors file(s)")

    # 2. Detect available modalities
    print("  🔍 Detecting modalities...")
    has_vision, has_audio, arch = detect_modalities(st_files)
    if arch == "phi4v":
        inner_prefix = _detect_phi4v_inner_prefix(st_files)
        print(f"  🏗️  Architecture: Phi4ForCausalLMV (vision_tower, inner: '{inner_prefix or 'none'}')")
    else:
        inner_prefix = "vision_model."
        if arch == "phi4mm":
            print(f"  🏗️  Architecture: Phi4MMForCausalLM (embed_tokens_extend)")
    print(f"  {'✅' if has_vision else '❌'} Vision tensors {'found' if has_vision else 'NOT found'}")
    print(f"  {'✅' if has_audio else '❌'} Audio tensors {'found' if has_audio else 'NOT found'}")

    if not has_vision and not has_audio:
        print(f"❌ No multimodal tensors found!", file=sys.stderr)
        sys.exit(1)

    # 3. Select mode
    if args.mode:
        mode = args.mode
        if mode == "vision" and not has_vision:
            print(f"❌ Vision mode requested but no vision tensors found", file=sys.stderr)
            sys.exit(1)
        if mode == "audio" and not has_audio:
            print(f"❌ Audio mode requested but no audio tensors found", file=sys.stderr)
            sys.exit(1)
        if mode == "omni" and not (has_vision and has_audio):
            if has_vision and not has_audio:
                print("⚠️  No audio tensors found (vision-only model) — falling back to vision mode.")
                mode = "vision"
            elif has_audio and not has_vision:
                print("⚠️  No vision tensors found — falling back to audio mode.")
                mode = "audio"
            else:
                print(f"❌ Omni mode requires both vision and audio tensors", file=sys.stderr)
                sys.exit(1)
    else:
        mode = show_menu(has_vision, has_audio)

    include_vision = mode in ("vision", "omni")
    include_audio  = mode in ("audio", "omni")

    # Auto-name output
    if args.output:
        output_path = args.output
    else:
        suffix = {"vision": "vision", "audio": "audio", "omni": "omni"}[mode]
        output_path = str(model_dir / f"mmproj-phi4-{suffix}-{args.dtype}.gguf")

    print(f"\n  Mode:     {mode}")
    print(f"  Output:   {output_path}")

    # 4. Get configs
    vision_cfg, audio_cfg = get_configs(model_dir, arch=arch)
    if include_vision:
        print(f"  Vision:   {vision_cfg['num_hidden_layers']} SigLIP layers, "
              f"{vision_cfg['image_size']}px, patch={vision_cfg['patch_size']}")
    if include_audio:
        print(f"  Audio:    {audio_cfg['num_hidden_layers']} Conformer layers, "
              f"{audio_cfg['num_mel_bins']} mel bins")

    # 5. Extract tensors
    print(f"\n🔍 Extracting {mode} tensors...")
    tensors, mapping_log, skipped = extract_tensors(
        st_files, include_vision, include_audio, output_dtype,
        arch=arch, inner_prefix=inner_prefix, vision_cfg=vision_cfg,
    )

    if not tensors:
        print(f"❌ No tensors extracted!", file=sys.stderr)
        sys.exit(1)

    v_count = sum(1 for n in tensors if n.startswith("v.") or n.startswith("mm.0") or n.startswith("mm.2"))
    a_count = sum(1 for n in tensors if n.startswith("a.") or n.startswith("mm.a"))
    print(f"  ✅ Extracted {len(tensors)} tensors "
          f"({v_count} vision, {a_count} audio)")

    if args.verbose:
        print(f"\n  Tensor mapping:")
        for orig, gguf in sorted(mapping_log.items()):
            print(f"    {orig}")
            print(f"      → {gguf}")

    # 6. Write GGUF
    print(f"\n📝 Writing mmproj GGUF...")
    writer = GGUFWriter(output_path)

    # General metadata
    writer.add_string("general.architecture", "clip")
    writer.add_string("general.type", "mmproj")
    mode_label = {"vision": "Vision", "audio": "Audio", "omni": "Omni (Vision+Audio)"}[mode]
    writer.add_string("general.name", f"Phi-4 Multimodal {mode_label} Projector")
    writer.add_uint32("general.file_type", 1 if args.dtype == "f16" else 0)

    # CLIP flags
    writer.add_bool("clip.has_vision_encoder", include_vision)
    writer.add_bool("clip.has_audio_encoder", include_audio)

    if include_vision:
        # Phi4ForCausalLMV uses projector_type "phi4" (MLP2x_GELU) per llama.cpp PR #20168
        _proj_type = "phi4" if arch == "phi4v" else "mlp"
        writer.add_string("clip.projector_type", _proj_type)
        if arch == "phi4v":
            writer.add_bool("clip.use_gelu", True)
        writer.add_uint32("clip.vision.image_size", vision_cfg["image_size"])
        writer.add_uint32("clip.vision.patch_size", vision_cfg["patch_size"])
        writer.add_uint32("clip.vision.embedding_length", vision_cfg["hidden_size"])
        writer.add_uint32("clip.vision.feed_forward_length", vision_cfg["intermediate_size"])
        writer.add_uint32("clip.vision.projection_dim", vision_cfg["projection_dim"])
        writer.add_uint32("clip.vision.block_count", vision_cfg["num_hidden_layers"])
        writer.add_uint32("clip.vision.attention.head_count", vision_cfg["num_attention_heads"])
        writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_cfg["layer_norm_eps"])
        writer.add_array("clip.vision.image_mean", SIGLIP_IMAGE_MEAN, GGUFValueType.FLOAT32)
        writer.add_array("clip.vision.image_std", SIGLIP_IMAGE_STD, GGUFValueType.FLOAT32)
        writer.add_bool("clip.use_gelu", True)

    if include_audio:
        writer.add_string("clip.audio.projector_type", "mlp")
        writer.add_uint32("clip.audio.num_mel_bins", audio_cfg["num_mel_bins"])
        writer.add_uint32("clip.audio.embedding_length", audio_cfg["hidden_size"])
        writer.add_uint32("clip.audio.feed_forward_length", audio_cfg["intermediate_size"])
        writer.add_uint32("clip.audio.projection_dim", audio_cfg["projection_dim"])
        writer.add_uint32("clip.audio.block_count", audio_cfg["num_hidden_layers"])
        writer.add_uint32("clip.audio.attention.head_count", audio_cfg["num_attention_heads"])
        writer.add_float32("clip.audio.attention.layer_norm_epsilon", audio_cfg["layer_norm_eps"])

    # Add tensors
    for name, data in sorted(tensors.items()):
        writer.add_tensor(name, data)

    writer.write()

    # 7. Print usage instructions
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    print_usage_instructions(mode, output_path, file_size_mb, len(tensors))


if __name__ == "__main__":
    main()
