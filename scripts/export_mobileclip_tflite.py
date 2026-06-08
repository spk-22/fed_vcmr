#!/usr/bin/env python3
"""
Export MobileCLIP-S1 text and image encoders to INT8 TFLite for Android.

Outputs:
  - checkpoints/export/mobileclip_text_encoder.tflite (~15 MB, INT8)
  - checkpoints/export/mobileclip_vision_encoder.tflite (~20 MB, INT8)
  - android/app/src/main/assets/mobileclip_vocab.json
  - android/app/src/main/assets/mobileclip_merges.txt

Usage:
  python scripts/export_mobileclip_tflite.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import numpy as np
import subprocess
import shutil
from pathlib import Path

# ============================================================================
# 1. Load MobileCLIP from HuggingFace / OpenCLIP
# ============================================================================

def get_mobileclip_models():
    """Download and cache MobileCLIP-S1 text and image encoders."""
    try:
        import open_clip
    except ImportError:
        print("Installing open_clip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "open-clip-torch"])
        import open_clip

    print("Loading MobileCLIP-S1 from OpenCLIP...")

    # Load model and tokenizer
    model_name = "mobileclip_s1"
    pretrained = "openai"

    try:
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device='cpu'
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as e:
        print(f"Error loading from OpenCLIP: {e}")
        print("Attempting alternative: hf_hub_download")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download("apple/MobileCLIP", "MobileCLIP-S1.pt", cache_dir="/tmp/mobileclip")
        model = torch.jit.load(model_path)
        transform = None
        tokenizer = None

    return model, tokenizer, transform


def extract_text_encoder(model):
    """Extract just the text encoding path from MobileCLIP."""
    # MobileCLIP has:
    #   model.token_embedding
    #   model.positional_embedding
    #   model.transformer
    #   model.text_projection (optional)
    #   model.ln_final

    class TextEncoderModule(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.token_embedding = clip_model.token_embedding
            self.positional_embedding = clip_model.positional_embedding
            self.transformer = clip_model.transformer
            self.ln_final = clip_model.ln_final

            # Determine if text projection exists
            if hasattr(clip_model, 'text_projection'):
                self.text_projection = clip_model.text_projection
            else:
                self.text_projection = None

        def forward(self, token_ids):
            """
            Input: token_ids [batch, 77] (int64)
            Output: embeddings [batch, 512] (float32)
            """
            x = self.token_embedding(token_ids)  # [batch, 77, 512]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # [77, batch, 512]
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # [batch, 77, 512]
            x = self.ln_final(x)

            # Use EOS token embedding as global representation
            x = x[:, -1, :]  # [batch, 512]

            if self.text_projection is not None:
                x = self.text_projection @ x

            return F.normalize(x, dim=-1)

    return TextEncoderModule(model)


def extract_vision_encoder(model):
    """Extract just the vision encoding path from MobileCLIP."""
    # MobileCLIP has:
    #   model.visual_encoder (ResNet or ViT backbone)
    #   model.vision_projection

    class VisionEncoderModule(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.visual = clip_model.visual
            if hasattr(clip_model, 'vision_projection'):
                self.vision_projection = clip_model.vision_projection
            else:
                self.vision_projection = None

        def forward(self, images):
            """
            Input: images [batch, 3, 224, 224] (float32, normalized)
            Output: embeddings [batch, 512] (float32)
            """
            x = self.visual(images)  # [batch, 512]

            if self.vision_projection is not None:
                x = self.vision_projection(x)

            return F.normalize(x, dim=-1)

    return VisionEncoderModule(model)


def export_text_encoder_to_tflite(text_encoder, out_dir="checkpoints/export"):
    """Export text encoder to TFLite INT8."""
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cpu')
    text_encoder = text_encoder.to(device).eval()

    # Quantization-aware training dummy run
    print("Exporting text encoder to TFLite...")

    # Create dummy input: token IDs [1, 77]
    dummy_input = torch.randint(0, 49407, (1, 77), dtype=torch.int32, device=device)

    # Convert to TorchScript
    try:
        scripted = torch.jit.trace(text_encoder, dummy_input)
    except:
        scripted = torch.jit.script(text_encoder)

    # Export to ONNX
    onnx_path = os.path.join(out_dir, "mobileclip_text_encoder.onnx")
    torch.onnx.export(
        text_encoder,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "token_ids": {0: "batch_size"},
            "embeddings": {0: "batch_size"}
        }
    )
    print(f"  → ONNX: {onnx_path}")

    # Convert ONNX to TFLite using onnx2tf
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx2tf"])

    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    tflite_dir = os.path.join(out_dir, "mobileclip_text_encoder_tflite")
    if os.path.exists(tflite_dir):
        shutil.rmtree(tflite_dir)

    # Use onnx2tf to convert
    result = subprocess.run([
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_dir,
        "-oiqt", "-q", "-nstqc"  # Output Int8 QuantizationTable, Quantize, No Save TF Check
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"onnx2tf warning: {result.stderr}")

    # Find and rename .tflite file
    tflite_files = list(Path(tflite_dir).glob("*.tflite"))
    if tflite_files:
        src = tflite_files[0]
        dst = os.path.join(out_dir, "mobileclip_text_encoder.tflite")
        shutil.move(str(src), dst)
        print(f"  → TFLite: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
        return dst
    else:
        print(f"ERROR: No .tflite file generated in {tflite_dir}")
        return None


def export_vision_encoder_to_tflite(vision_encoder, out_dir="checkpoints/export"):
    """Export vision encoder to TFLite INT8."""
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cpu')
    vision_encoder = vision_encoder.to(device).eval()

    print("Exporting vision encoder to TFLite...")

    # Create dummy input: images [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    # Export to ONNX
    onnx_path = os.path.join(out_dir, "mobileclip_vision_encoder.onnx")
    torch.onnx.export(
        vision_encoder,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "embeddings": {0: "batch_size"}
        }
    )
    print(f"  → ONNX: {onnx_path}")

    # Convert ONNX to TFLite using onnx2tf
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    tflite_dir = os.path.join(out_dir, "mobileclip_vision_encoder_tflite")
    if os.path.exists(tflite_dir):
        shutil.rmtree(tflite_dir)

    result = subprocess.run([
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_dir,
        "-oiqt", "-q", "-nstqc"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"onnx2tf warning: {result.stderr}")

    # Find and rename .tflite file
    tflite_files = list(Path(tflite_dir).glob("*.tflite"))
    if tflite_files:
        src = tflite_files[0]
        dst = os.path.join(out_dir, "mobileclip_vision_encoder.tflite")
        shutil.move(str(src), dst)
        print(f"  → TFLite: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
        return dst
    else:
        print(f"ERROR: No .tflite file generated in {tflite_dir}")
        return None


def export_tokenizer_assets(tokenizer, out_dir="android/app/src/main/assets"):
    """Export BPE vocab and merges for Java tokenizer."""
    os.makedirs(out_dir, exist_ok=True)

    print("Exporting tokenizer assets...")

    if tokenizer is not None:
        # OpenCLIP's tokenizer has a vocab attribute
        vocab_dict = tokenizer.vocab
        merges_list = tokenizer.merges if hasattr(tokenizer, 'merges') else []
    else:
        # Fallback: use standard CLIP vocab
        # Download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            vocab_path = hf_hub_download("openai/clip-vit-base-patch32", "vocab.json")
            merges_path = hf_hub_download("openai/clip-vit-base-patch32", "merges.txt")

            with open(vocab_path) as f:
                vocab_dict = json.load(f)
            with open(merges_path) as f:
                merges_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error downloading CLIP tokenizer: {e}")
            print("Using minimal fallback vocab (not recommended for production)")
            vocab_dict = {}
            merges_list = []

    # Save vocab.json
    vocab_out = os.path.join(out_dir, "mobileclip_vocab.json")
    with open(vocab_out, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"  → vocab: {vocab_out}")

    # Save merges.txt
    merges_out = os.path.join(out_dir, "mobileclip_merges.txt")
    with open(merges_out, 'w') as f:
        for merge in merges_list:
            f.write(merge + '\n')
    print(f"  → merges: {merges_out}")


def verify_export(text_tflite_path, vision_tflite_path):
    """Quick validation that models load and produce reasonable outputs."""
    print("\nVerifying exported models...")

    try:
        import tensorflow as tf

        # Load interpreters
        text_interp = tf.lite.Interpreter(text_tflite_path)
        text_interp.allocate_tensors()

        vision_interp = tf.lite.Interpreter(vision_tflite_path)
        vision_interp.allocate_tensors()

        # Test text encoder
        text_input = text_interp.get_input_details()[0]
        text_output = text_interp.get_output_details()[0]

        dummy_tokens = np.random.randint(0, 49407, (1, 77), dtype=np.int32)
        text_interp.set_tensor(text_input['index'], dummy_tokens)
        text_interp.invoke()
        text_emb = text_interp.get_tensor(text_output['index'])
        print(f"  ✓ Text encoder output: {text_emb.shape}")

        # Test vision encoder
        vision_input = vision_interp.get_input_details()[0]
        vision_output = vision_interp.get_output_details()[0]

        dummy_images = np.random.randn(1, 3, 224, 224).astype(np.float32)
        vision_interp.set_tensor(vision_input['index'], dummy_images)
        vision_interp.invoke()
        vision_emb = vision_interp.get_tensor(vision_output['index'])
        print(f"  ✓ Vision encoder output: {vision_emb.shape}")

    except Exception as e:
        print(f"  Warning: Could not verify: {e}")


def main():
    out_dir = "checkpoints/export"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("MobileCLIP-S1 TFLite Export for Android")
    print("=" * 70)

    # Load models
    try:
        model, tokenizer, transform = get_mobileclip_models()
    except Exception as e:
        print(f"ERROR: Could not load MobileCLIP: {e}")
        print("\nNote: This script requires:")
        print("  pip install open-clip-torch huggingface_hub")
        print("  pip install onnx2tf tensorflow")
        return 1

    # Extract encoders
    text_encoder = extract_text_encoder(model)
    vision_encoder = extract_vision_encoder(model)

    # Export to TFLite
    text_tflite = export_text_encoder_to_tflite(text_encoder, out_dir)
    vision_tflite = export_vision_encoder_to_tflite(vision_encoder, out_dir)

    # Export tokenizer
    export_tokenizer_assets(tokenizer, "android/app/src/main/assets")

    # Verify
    if text_tflite and vision_tflite:
        verify_export(text_tflite, vision_tflite)

    print("\n" + "=" * 70)
    print("Export complete!")
    print("Next: Implement TextEncoder.java and VisionEncoder.java")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
