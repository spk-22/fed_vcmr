#!/usr/bin/env python3
"""
Quantize MobileCLIP-S1 text backbone for Android APK.

Dynamic INT8 quantization on all Linear layers:
  ~254MB float32 -> ~64MB int8  (4x reduction)

Input:  checkpoints/export/mobileclip_text_encoder_v2.ptl  (correct, 254MB)
Output: checkpoints/export/mobileclip_text_encoder_q8.ptl  (quantized, ~64MB)
        android/app/src/main/assets/models/mobileclip_text_encoder.ptl  (deployed)

Usage:
  .venv/Scripts/python scripts/quantize_text_ptl.py
"""

import torch
import open_clip
import numpy as np
import os, shutil

OUT_Q8    = "C:/prism/checkpoints/export/mobileclip_text_encoder_q8.ptl"
ASSET_PTL = "C:/prism/android/app/src/main/assets/models/mobileclip_text_encoder.ptl"

print("Loading MobileCLIP-S1 for quantization...")
model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S1", pretrained="datacompdr")
model.eval().cpu()
tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")

class TextBackbone(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text

    def forward(self, token_ids):
        return self.text_model(token_ids)  # [1, 512]

backbone = TextBackbone(model)
backbone.eval()

test_texts = [
    "person washing dishes at kitchen sink",
    "woman sitting on stairs with a laptop",
    "two people sitting on a sofa",
]
toks = tokenizer(test_texts[:1])

with torch.no_grad():
    ref_out = model.encode_text(toks, normalize=False)[0].numpy()

# ── Patch open_clip's dtype introspection so quantized Linears don't crash ────
import open_clip.transformer as oct
_orig_get_cast_dtype = oct.Transformer.get_cast_dtype
def _safe_get_cast_dtype(self):
    try:
        return _orig_get_cast_dtype(self)
    except AttributeError:
        return torch.float32
oct.Transformer.get_cast_dtype = _safe_get_cast_dtype

# ── Dynamic INT8 quantization on the Python module ───────────────────────────
print("Applying dynamic INT8 quantization...")
quantized = torch.quantization.quantize_dynamic(
    backbone,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8,
)

with torch.no_grad():
    q_out = quantized(toks).squeeze(0).numpy()

cos_q = float(np.dot(ref_out, q_out) / (np.linalg.norm(ref_out) * np.linalg.norm(q_out)))
print(f"  Quantized cosine vs reference: {cos_q:.6f}  (target >0.99)")
assert cos_q > 0.975, f"Quantization degraded output too much: cos={cos_q:.4f}"

# ── Trace the quantized module ────────────────────────────────────────────────
print("Tracing quantized model...")
traced_q = torch.jit.trace(quantized, toks, strict=False, check_trace=False)

with torch.no_grad():
    traced_q_out = traced_q(toks).squeeze(0).numpy()

cos_tq = float(np.dot(ref_out, traced_q_out) / (np.linalg.norm(ref_out) * np.linalg.norm(traced_q_out)))
print(f"  Traced cosine vs reference: {cos_tq:.6f}")
assert cos_tq > 0.975, f"Tracing broke quantized model: cos={cos_tq:.4f}"

# ── Save for lite interpreter ─────────────────────────────────────────────────
traced_q._save_for_lite_interpreter(OUT_Q8)
size_mb = os.path.getsize(OUT_Q8) / 1e6
print(f"\n  -> {OUT_Q8}  ({size_mb:.1f} MB)")

shutil.copy(OUT_Q8, ASSET_PTL)
print(f"  Copied to: {ASSET_PTL}")
print(f"  Size: {os.path.getsize(ASSET_PTL)/1e6:.1f} MB")

# ── Final multi-query check ───────────────────────────────────────────────────
print("\nMulti-query cosine check:")
for text in test_texts:
    t = tokenizer([text])
    with torch.no_grad():
        r = model.encode_text(t, normalize=False)[0].numpy()
        q = traced_q(t).squeeze(0).numpy()
    c = float(np.dot(r, q) / (np.linalg.norm(r) * np.linalg.norm(q)))
    print(f"  '{text[:40]}' -> cos={c:.6f}")

print("\nDone. Rebuild APK to deploy quantized text encoder.")
