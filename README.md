# FedVCMR: Federated Video-Caption Matching and Retrieval

A Windows-native implementation of zero-shot video retrieval using MobileCLIP-S1 and FAISS.

## Architecture
- **Backbone**: MobileCLIP-S1 (via `open-clip`).
- **Storage**: SQLite for metadata, `.npy` for disk-based feature caching.
- **Search**: FAISS (Inner Product) + MaxSim Reranking.
- **Optimization**: Producer-consumer pattern with multi-threaded PyAV decoding and CUDA-accelerated inference.

## Prerequisites
- Windows OS
- Python 3.10+
- FFmpeg (for PyAV)
- NVIDIA GPU (Optional, for CUDA acceleration)

## Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/spk-22/fed_vcmr.git
   cd fed_vcmr
   ```

2. **Create Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   # For CUDA support (recommended):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Data Preparation**:
   - Download the MSR-VTT dataset.
   - Place videos in `c:\prism\MSRVTT\MSRVTT\videos\all`.
   - Place `MSR_VTT.json` in `c:\prism\MSRVTT\MSRVTT\annotation\`.

## Usage

### 1. Ingestion
Process videos, generate chunks, and build the feature cache/index:
```powershell
python -m src.ingestion
```

### 2. Smoke Test
Verify the backbone and GPU acceleration:
```powershell
python smoke_test.py
```

### 3. Evaluation
Run the 1K-A zero-shot evaluation:
```powershell
python eval_1ka.py
```

## Milestones
- [x] M1: MobileCLIP-S1 Smoke Test
- [x] M2: Initial Ingestion Pipe (7 Videos)
- [x] M3: Basic Retrieval Service (MaxSim)
- [x] M4: Full MSR-VTT Pipeline (Optimized & Resumable)
- [x] M5: Zero-Shot Evaluation (1K-A)
