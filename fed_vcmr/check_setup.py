import torch
import av
import os
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import VIDEO_ROOT, ANNOTATION_PATH, BACKBONE_NAME
from src.backbone import MobileCLIPWrapper

def check_setup():
    print("=== System Check ===")

    # 1. Check Torch & CUDA
    print(f"Torch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # 2. Check Data Paths
    print(f"\nChecking Data Paths:")
    print(f"Video Root: {VIDEO_ROOT}")
    print(f"Exists: {VIDEO_ROOT.exists()}")

    print(f"Annotation Path: {ANNOTATION_PATH}")
    print(f"Exists: {ANNOTATION_PATH.exists()}")

    if not VIDEO_ROOT.exists() or not ANNOTATION_PATH.exists():
        print("ERROR: Data paths not found.")
        return

    # 3. Check specific video
    sample_video = list(VIDEO_ROOT.glob("*.mp4"))
    if not sample_video:
        print("ERROR: No .mp4 files found in Video Root.")
    else:
        v = sample_video[0]
        print(f"Found {len(sample_video)} videos.")
        print(f"Testing PyAV on {v.name}...")
        try:
            container = av.open(str(v))
            print(f"  Duration: {float(container.duration) / 1000000.0}s")
            print(f"  Streams: {container.streams.video}")
            container.close()
        except Exception as e:
            print(f"  PyAV Error: {e}")

    # 4. Check Backbone Loading
    print(f"\nLoading Backbone ({BACKBONE_NAME})...")
    try:
        model = MobileCLIPWrapper()
        print("  Model loaded successfully.")

        # Test inference
        dummy_img = Image.new('RGB', (224, 224), color='red')
        emb = model.encode_images([dummy_img])
        print(f"  Image Encoding Shape: {emb.shape}")

    except Exception as e:
        print(f"  Model Load Error: {e}")

    print("\n=== Check Complete ===")

if __name__ == "__main__":
    check_setup()

