#!/usr/bin/env python3
"""
Export MobileCLIP-S1 vision backbone with fixed batch=8 for fast ingest.

Produces: checkpoints/export/mobileclip_vision_encoder_b8.tflite
  Input:  [8, 256, 256, 3]  float32 NHWC
  Output: [8, 512]          float32

On Android, VisionEncoder runs 4 invocations of 8 frames instead of 32 of 1.
~4x faster ingest with identical embeddings.

Usage:
  .venv/Scripts/python scripts/export_vision_batch.py
"""

import os, sys, shutil, json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

BATCH      = 8
IMG_SIZE   = 256
EMBED_DIM  = 512
OUT_DIR    = "C:/prism/checkpoints/export"
ASSET_DIR  = "C:/prism/android/app/src/main/assets/models"
ONNX_PATH  = f"{OUT_DIR}/mobileclip_vision_encoder_b{BATCH}.onnx"
TFLITE_OUT = f"{OUT_DIR}/mobileclip_vision_encoder_b{BATCH}.tflite"

# ── 1. Load MobileCLIP-S1 ────────────────────────────────────────────────────

print("Loading MobileCLIP-S1 via open_clip...")
import open_clip
model, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr')
model.eval().cpu()
print("  ok")

# ── 2. Wrap visual backbone (no projection head — head is a separate TFLite) ─

class VisualBackbone(nn.Module):
    """Returns raw visual features before the projection head."""
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x):           # x: [B, 3, H, W]
        return self.visual(x)       # [B, EMBED_DIM]

backbone = VisualBackbone(model)

# ── 3. Verify output dim matches existing deployed model ─────────────────────

dummy = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    out = backbone(dummy)
print(f"  Backbone output: {out.shape}  (expected [{BATCH}, {EMBED_DIM}])")
assert out.shape == (BATCH, EMBED_DIM), f"Unexpected output shape: {out.shape}"

# ── 4. Export to ONNX with FIXED batch=8 ────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
print(f"\nExporting ONNX (fixed batch={BATCH}, size={IMG_SIZE})...")
torch.onnx.export(
    backbone,
    dummy,
    ONNX_PATH,
    opset_version=14,
    input_names=["images"],
    output_names=["features"],
    # NO dynamic_axes — fixed batch so TFLite shape is locked to [8,256,256,3]
)
import onnx
m = onnx.load(ONNX_PATH)
onnx.checker.check_model(m)
print(f"  → {ONNX_PATH}  ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")

# ── 5. Convert ONNX → TFLite via onnx2tf ────────────────────────────────────

tflite_tmp = f"{OUT_DIR}/mobileclip_vision_b{BATCH}_tflite"
if os.path.exists(tflite_tmp): shutil.rmtree(tflite_tmp)

print(f"\nConverting to TFLite (onnx2tf)...")
import subprocess
r = subprocess.run([
    sys.executable, "-m", "onnx2tf",
    "-i", ONNX_PATH,
    "-o", tflite_tmp,
    "--non_verbose",
], capture_output=True, text=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)

# Find the .tflite file
tflites = list(Path(tflite_tmp).glob("*.tflite"))
if not tflites:
    print("ERROR: no .tflite produced"); sys.exit(1)

# Prefer float32 (not INT8) for embedding accuracy
fp_tflites = [f for f in tflites if "quant" not in f.name.lower()]
chosen = fp_tflites[0] if fp_tflites else tflites[0]
shutil.copy(str(chosen), TFLITE_OUT)
print(f"  → {TFLITE_OUT}  ({os.path.getsize(TFLITE_OUT)/1e6:.1f} MB)")

# ── 6. Verify output matches single-frame model ──────────────────────────────

print("\nVerifying batch model output...")
import tensorflow as tf

interp_b8 = tf.lite.Interpreter(TFLITE_OUT)
interp_b8.allocate_tensors()
inp_det  = interp_b8.get_input_details()[0]
out_det  = interp_b8.get_output_details()[0]
print(f"  Input:  {inp_det['shape']}")
print(f"  Output: {out_det['shape']}")

# Compare against single-frame model on same input
interp_b1 = tf.lite.Interpreter(
    f"{ASSET_DIR}/mobileclip_vision_encoder.tflite")
interp_b1.allocate_tensors()

test_img = np.random.randn(1, 256, 256, 3).astype(np.float32)

interp_b1.set_tensor(interp_b1.get_input_details()[0]['index'], test_img)
interp_b1.invoke()
emb_b1 = interp_b1.get_tensor(interp_b1.get_output_details()[0]['index'])[0]

batch_input = np.tile(test_img, (BATCH, 1, 1, 1))
interp_b8.set_tensor(inp_det['index'], batch_input)
interp_b8.invoke()
emb_b8 = interp_b8.get_tensor(out_det['index'])[0]

cos = float(np.dot(emb_b1, emb_b8) / (np.linalg.norm(emb_b1) * np.linalg.norm(emb_b8)))
print(f"  Cosine similarity b1 vs b8 on same frame: {cos:.6f}  (target: >0.999)")
if cos < 0.99:
    print("  WARNING: outputs diverged — may need calibration")
else:
    print("  OK")

# ── 7. Copy to Android assets ────────────────────────────────────────────────

dst = f"{ASSET_DIR}/mobileclip_vision_encoder_b8.tflite"
shutil.copy(TFLITE_OUT, dst)
print(f"\nCopied to Android assets: {dst}")
print(f"  ({os.path.getsize(dst)/1e6:.1f} MB)")

print("\nDone. Update VisionEncoder.java to use mobileclip_vision_encoder_b8.tflite")
print("for encodeBatch() calls during ingest.")
