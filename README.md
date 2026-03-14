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

---

## Complete Project Milestones

### Phase 1 — Working Retrieval (No Training)

| Milestone | What You Build                                                 | Gate                                   |
| --------- | -------------------------------------------------------------- | -------------------------------------- |
| **M1**    | MobileCLIP-S1 loads, encodes image + text → (1, 512) confirmed | Shape correct, latency <60ms           |
| **M2**    | Frame extractor + fixed chunker + FAISS index on 7 test videos | Index builds, SQLite populated         |
| **M3**    | Text query → FAISS → MaxSim rerank → timestamped results       | ≥3/5 test queries return correct video |

### Phase 2 — Real Data + Baseline Numbers

| Milestone | What You Build                                                             | Gate                          |
| --------- | -------------------------------------------------------------------------- | ----------------------------- |
| **M4**    | Full MSR-VTT indexed (6,513 videos, backbone cache as mmap)                | Cache built in <90 min        |
| **M5**    | Zero-shot evaluation on MSR-VTT 1K-A test split                            | R@1 > 20%                     |
| **M6**    | ActivityNet annotations parsed, 200 videos downloaded, eval pipeline ready | IoU metric computes correctly |
| **M7**    | Zero-shot moment retrieval on ActivityNet val subset                       | R@1 IoU=0.5 > 5%              |

### Phase 3 — Training

| Milestone | What You Build                                            | Gate                                     |
| --------- | --------------------------------------------------------- | ---------------------------------------- |
| **M8**    | Projection heads (Vision + Text, 512→256) + InfoNCE loss  | Loss drops from ~4.85 within 100 steps   |
| **M9**    | THNC memory bank curriculum (3-phase easy→medium→hard)    | Hard negative cosine > 0.8 confirmed     |
| **M10**   | Centralized training on 130K MSR-VTT pairs (30 epochs)    | MSR-VTT R@1 improves >+5 over zero-shot  |
| **M11**   | DGSE dual-granularity pooling integrated into reranking   | ActivityNet R@1 IoU=0.7 improves over M7 |
| **M12**   | Moment losses (L_MR + L_VR + L_MaxSim + L_Temporal) added | ActivityNet R@1 IoU=0.7 > 6%             |

### Phase 4 — Cross-Modal Transformer

| Milestone | What You Build                                               | Gate                            |
| --------- | ------------------------------------------------------------ | ------------------------------- |
| **M13**   | 2-layer, 4-head, 256-dim Transformer implemented             | Forward pass runs without error |
| **M14**   | Centrally trained on ActivityNet train subset (strict split) | Boundary loss decreases         |
| **M15**   | Transformer frozen + integrated into full inference pipeline | End-to-end latency <100ms       |
| **M16**   | ActivityNet R@1 IoU=0.7 evaluated with full pipeline         | Target: beat MA-VR 10.04%       |

### Phase 5 — Federated Learning

| Milestone | What You Build                                         | Gate                                          |
| --------- | ------------------------------------------------------ | --------------------------------------------- |
| **M17**   | MSR-VTT partitioned into 5 non-IID clients by category | Each client has ~26K pairs                    |
| **M18**   | FedProx local training + DP-SGD + gradient clipping    | Weight delta ~4MB per round                   |
| **M19**   | Trimmed Mean aggregation + uniformity circuit breaker  | Circuit breaker triggers on injected collapse |
| **M20**   | DAPHW domain adapter (local only, never federated)     | Adapter weights never in server delta         |
| **M21**   | 50 FL rounds simulated, R@1 vs round curve plotted     | FL R@1 ≥ 85% of centralized R@1               |
| **M22**   | Byzantine robustness test (1 noisy client injected)    | FL with noisy client still beats zero-shot    |
| **M23**   | Index rebuild after FL round via backbone cache        | Rebuild completes <1 second                   |

### Phase 6 — Ablations + Paper Numbers

| Milestone | What You Build                                                     | Gate                                              |
| --------- | ------------------------------------------------------------------ | ------------------------------------------------- |
| **M24**   | Ablation Table 1: MA-VR backbone + your architecture vs yours      | Row 2 > Row 1 confirms architectural contribution |
| **M25**   | Ablation Table 2: DGSE → THNC → DAPHW → Transformer additive       | Each row shows positive gain                      |
| **M26**   | Ablation Table 3: FedAvg vs FedProx vs Byzantine                   | FedProx > FedAvg confirmed                        |
| **M27**   | Ablation Table 4: 1-caption vs 20-caption training                 | 20-caption significantly better                   |
| **M28**   | Final numbers table: zero-shot → centralized → FL on both datasets | All numbers reproducible                          |

### Phase 7 — Samsung Deployment (Optional for Paper, Required for Full System)

| Milestone | What You Build                                                         | Gate                                       |
| --------- | ---------------------------------------------------------------------- | ------------------------------------------ |
| **M29**   | MobileCLIP-S1 + Transformer quantized to INT8                          | Accuracy drop <2% vs FP32                  |
| **M30**   | ONNX export + SNPE DLC conversion for Hexagon NPU                      | Model runs on Android emulator             |
| **M31**   | Android app: FAISS JNI + Room database + query UI                      | End-to-end demo on physical Samsung device |
| **M32**   | FL client on Android: WorkManager + thermal check + secure aggregation | FL round completes on device               |

