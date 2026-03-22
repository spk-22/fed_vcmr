import os
from pathlib import Path
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "MSRVTT"
VIDEO_ROOT = DATA_ROOT / "videos" / "all"
ANNOTATION_PATH = DATA_ROOT / "annotation" / "MSR_VTT.json"
CACHE_ROOT = PROJECT_ROOT / "cache"

# Database
DB_PATH = PROJECT_ROOT / "fedvcmr.db"

# Model Configuration
BACKBONE_NAME = "MobileCLIP-S1"
PRETRAINED_TAG = "datacompdr"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = "float16"
USE_PROJECTION = True  # Set to True to enable trainable projection head
USE_ADAPTER = True     # Set to True to enable client-specific AdapterHead

# Ingestion Policy
CHUNK_DURATION = 8.0  # seconds
CHUNK_STRIDE = 4.0    # seconds
FRAMES_PER_CHUNK = 8

# FAISS Configuration
INDEX_PATH = PROJECT_ROOT / "faiss_index.bin"
DIMENSION = 512

# Create necessary directories
os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT / "frame_features", exist_ok=True)
