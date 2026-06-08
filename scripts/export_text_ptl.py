#!/usr/bin/env python3
"""
Re-export MobileCLIP-S1 text backbone to PyTorch Mobile (.ptl).

Accepts: int64 token_ids [1, 77]
Returns: float32 raw features [512]  (NOT normalized — TextEncoder.java normalizes after head)

Usage:
  .venv/Scripts/python scripts/export_text_ptl.py
"""

import torch
import open_clip
import numpy as np

OUT_PTL   = "C:/prism/checkpoints/export/mobileclip_text_encoder_v2.ptl"
ASSET_PTL = "C:/prism/android/app/src/main/assets/models/mobileclip_text_encoder.ptl"

print("Loading MobileCLIP-S1...")
model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S1", pretrained="datacompdr")
model.eval().cpu()
tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")

class TextBackbone(torch.nn.Module):
    """Wraps MobileCLIP-S1 CustomTextCLIP text encoder: token_ids [1,77] -> raw features [512]."""
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text  # TextTransformer

    def forward(self, token_ids):          # [1, 77] int64
        return self.text_model(token_ids)  # [1, 512]  (pooled, no normalize)

backbone = TextBackbone(model)
backbone.eval()

# ── Verify against open_clip reference ───────────────────────────────────────
test_text = "person washing dishes at kitchen sink"
toks = tokenizer([test_text])

with torch.no_grad():
    ref_out = model.encode_text(toks, normalize=False)[0].numpy()
    new_out = backbone(toks).squeeze(0).numpy()

max_diff = float(np.max(np.abs(ref_out - new_out)))
cos = float(np.dot(ref_out, new_out) / (np.linalg.norm(ref_out) * np.linalg.norm(new_out)))
print(f"Reference[:8]: {ref_out[:8]}")
print(f"Backbone[:8]:  {new_out[:8]}")
print(f"Max diff: {max_diff:.6f}  Cosine: {cos:.6f}")
assert cos > 0.999, f"Backbone mismatch! cos={cos:.4f}"
print("  Backbone verified OK")

# ── Script and export for PyTorch Mobile ─────────────────────────────────────
print("\nExporting to PyTorch Mobile...")
traced = torch.jit.trace(backbone, toks, strict=False, check_trace=False)

# Verify traced version
with torch.no_grad():
    traced_out = traced(toks).squeeze(0).numpy()
cos_traced = float(np.dot(ref_out, traced_out) / (np.linalg.norm(ref_out) * np.linalg.norm(traced_out)))
print(f"Traced cosine vs reference: {cos_traced:.6f}")
assert cos_traced > 0.999, f"Tracing broke it! cos={cos_traced:.4f}"

traced._save_for_lite_interpreter(OUT_PTL)
print(f"  -> {OUT_PTL}")

# ── Copy to Android assets ────────────────────────────────────────────────────
import shutil, os
shutil.copy(OUT_PTL, ASSET_PTL)
print(f"  Copied to: {ASSET_PTL}")
print(f"  Size: {os.path.getsize(ASSET_PTL)/1e6:.1f} MB")
print("\nDone. Rebuild APK to pick up new text encoder.")
