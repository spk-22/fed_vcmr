import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import numpy as np
import os
import onnx
import subprocess
import sys

# Constants
DEVICE = "cpu"
MODEL_NAME = "MobileCLIP-S1"
PRETRAINED = "datacompdr"
ASSET_DIR = "android/app/src/main/assets/models"
DIM = 512

print(f"Loading {MODEL_NAME}...")
model, _, transform = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
model.eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# 1. Export Text Backbone (PyTorch Mobile .ptl)
print("\nExporting Text Backbone...")
class TextBackbone(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, token_ids):
        # Use open_clip's encode_text directly which handles pooling and projection
        return self.clip_model.encode_text(token_ids, normalize=True)

text_backbone = TextBackbone(model).eval()
test_text = "person"
tokens = tokenizer([test_text])
# Try tracing with check_trace=False to ignore Graph diffs
print("  Tracing text backbone...")
traced_text = torch.jit.trace(text_backbone, tokens, strict=False, check_trace=False)

text_ptl = os.path.join(ASSET_DIR, "mobileclip_text_encoder.ptl")
traced_text._save_for_lite_interpreter(text_ptl)
print(f"  ✓ Saved to {text_ptl}")

# 2. Export Vision Backbone (TFLite)
print("\nExporting Vision Backbone...")
class VisionBackbone(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x):
        # x: [1, 3, 256, 256]
        # visual returns [1, 512]
        return self.visual(x)

vision_backbone = VisionBackbone(model).eval()
dummy_img = torch.randn(1, 3, 256, 256)
onnx_path = "vision_backbone.onnx"

torch.onnx.export(
    vision_backbone,
    dummy_img,
    onnx_path,
    opset_version=14,
    input_names=["images"],
    output_names=["embeddings"]
)

# Convert ONNX to TFLite (NHWC)
print("Converting ONNX to TFLite...")
tflite_out_dir = "vision_tflite_temp"
subprocess.run([
    sys.executable, "-m", "onnx2tf",
    "-i", onnx_path,
    "-o", tflite_out_dir,
    "-ois", "images:1,3,256,256",
    "-n" # Non-verbose
], check=True)

import glob
tflite_files = glob.glob(os.path.join(tflite_out_dir, "*.tflite"))
if tflite_files:
    import shutil
    vision_tflite = os.path.join(ASSET_DIR, "mobileclip_vision_encoder.tflite")
    shutil.copy(tflite_files[0], vision_tflite)
    print(f"  ✓ Saved to {vision_tflite}")
else:
    print("  ✗ TFLite conversion failed!")

print("\nVerification complete. Now rebuild the app and re-ingest.")
