import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "MSRVTT" / "MSRVTT"
VIDEO_ROOT = DATA_ROOT / "videos" / "all"
ANNOTATION_PATH = DATA_ROOT / "annotation" / "MSR_VTT.json"
CACHE_ROOT = PROJECT_ROOT / "cache"

ACTIVITYNET_DATA_ROOT = PROJECT_ROOT / "data" / "activitynet"
ACTIVITYNET_VIDEO_ROOT = ACTIVITYNET_DATA_ROOT / "videos"
ACTIVITYNET_ANNOTATION_PATH = ACTIVITYNET_DATA_ROOT / "annotations.json"


# Database
DB_PATH = PROJECT_ROOT / "fedvcmr.db"

# Model Configuration
BACKBONE_NAME = "MobileCLIP-S1"
PRETRAINED_TAG = "datacompdr"
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float16" if DEVICE == "cuda" else "float32"

# Ingestion Policy
CHUNK_DURATION = 8.0  # seconds
CHUNK_STRIDE = 4.0    # seconds
FRAMES_PER_CHUNK = 8

# FAISS Configuration
INDEX_PATH = PROJECT_ROOT / "faiss_index.bin"
DIMENSION = 512

# Transformer Configuration (Phase 4)
TRANSFORMER_CHECKPOINT_PATH = CACHE_ROOT / "transformer_best.pt"
TRANSFORMER_EMBED_DIM = 256
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2

# Create necessary directories
os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT / "frame_features", exist_ok=True)
