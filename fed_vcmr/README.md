# FedVCMR: Federated Video-Caption Matching and Retrieval (Phase 1)

We have successfully implemented the first phase of the Federated Video-Caption Matching and Retrieval (FedVCMR) system. The system is Windows-native, optimized for local hardware (GPU/CUDA), and verified against the MSR-VTT 1K-A benchmark.

## Accomplishments

### 1. High-Performance Ingestion Pipe
- **Producer-Consumer Pattern**: Multi-threaded decoding using PyAV to parallelize the heavy CPU task of frame extraction.
- **CUDA Acceleration**: Configured a CUDA-enabled environment for the MobileCLIP-S1 backbone, reducing frame encoding latency from seconds to **~63ms**.
- **Resumable State**: Used SQLite to track processed videos and chunks, allowing the pipeline to resume seamlessly.
- **Smart Chunking**: Implemented center-biased temporal weights for chunk generation as per the technical reference.

### 2. Retrieval Service (M3/M4)
- **FAISS Integration**: Efficient coarse search using normalized Inner Product (IP) index.
- **MaxSim Reranking**: Vectorized implementation of the MaxSim similarity metric for high-accuracy local ranking.
- **Query Service**: Rule-based expansion and batched encoding to prevent VRAM overflow.

### 3. Quantitative Evaluation (M5)
The system was validated against the **MSR-VTT 1K-A** test split (1,000 videos). 

| Metric | Result (%) | Goal |
| :--- | :--- | :--- |
| **R@1** | **25.80** | > 20% |
| **R@5** | **49.20** | - |
| **R@10** | **59.10** | - |
| **Median Rank** | **6.0** | - |

> [!NOTE]
> These results were achieved in a **Zero-Shot** setting using the MobileCLIP-S1 backbone.

---

## Setup & Usage

### Prerequisites
- Windows OS
- Python 3.10+
- FFmpeg (for PyAV)
- NVIDIA GPU (MX250+ recommended for CUDA)

### Installation
1. **Clone and Setup**:
   ```powershell
   git clone https://github.com/spk-22/fed_vcmr.git
   cd fed_vcmr
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Dependencies**:
   ```powershell
   pip install -r requirements.txt
   # For CUDA support:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Data Preparation
- Videos: `c:\prism\MSRVTT\MSRVTT\videos\all`
- Annotations: `c:\prism\MSRVTT\MSRVTT\annotation\MSR_VTT.json`

### Run
```powershell
# Ingestion
python -m src.ingestion

# Zero-Shot Eval
python eval_1ka.py
```

## Data Artifacts
- **Metadata**: `fedvcmr.db` (SQLite)
- **Feature Cache**: `c:\prism\cache\frame_features\` (24,000+ `.npy` files)
