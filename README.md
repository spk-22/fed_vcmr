# FedVCMR: Federated Video Corpus Moment Retrieval

FedVCMR is a distributed, privacy-preserving video search engine designed to run on resource-constrained edge devices (like the Samsung Galaxy A55). It utilizes a multi-stage retrieval pipeline with a MobileCLIP backbone and a Cross-Modal Transformer for precise temporal grounding.

---

## 🚀 Status: Milestone M10 Passed
Current R@1 on MSR-VTT 1K-A: **31.2%** (+5.4% improvement over zero-shot baseline).

---

## 🗺️ Project Milestones (1–32)

### Phase 1: Working Retrieval (Base Infrastructure)
- [x] **M1: Core VLM Setup** — MobileCLIP-S1 integration, encoding image + text (512-d).
- [x] **M2: Chunking Engine** — Frame extractor + sliding window (8s duration/4s stride).
- [x] **M3: Minimal Search** — SQLite metadata storage + basic FAISS query pipeline.

### Phase 2: Real Data & Baselines
- [x] **M4: Mass Ingestion** — MSR-VTT 10k indexed (6,513 videos), binary feature cache as memmap.
- [x] **M5: Zero-Shot Baseline** — Evaluation on MSR-VTT 1K-A test split (Target: >20%, Achieved: 25.8%).
- [x] **M6: ActivityNet Support** — ActivityNet annotations parsed, 200 videos downloaded, eval ready.
- [x] **M7: Moment Evaluation** — Zero-shot moment retrieval on ActivityNet val subset.

### Phase 3: Centralized Training
- [x] **M8: Projection Learning** — Linear projection heads (512→512) + InfoNCE loss gradient flow.
- [x] **M9: Dataset Alignment** — Fixed `chunk_id` sort order mismatch between .bin cache and training loop.
- [x] **M10: Centralized Gate** — R@1 performance on MSR-VTT 1K-A exceeds baseline by >5% (Achieved: 31.2%).
- [ ] **M11: DGSE Integration** — Dual-granularity pooling for better frame-level late interaction.
- [ ] **M12: Moment Loss** — Joint optimization of MR + VR + Temporal boundary losses.

### Phase 4: Cross-Modal Transformer
- [ ] **M13: Transformer Arch** — 2-layer, 4-head, 256-dim Transformer implementation.
- [ ] **M14: Temporal Grounding** — Contextualized frame encoding for precise boundary prediction.
- [ ] **M15: Pipeline Integration** — Merging Transformer into the end-to-end inference loop.
- [ ] **M16: Landmark Benchmarking** — Target: Beat MA-VR (10.04% R@1 IoU=0.7) on ActivityNet.

### Phase 5: Federated Learning (FL)
- [ ] **M17: Client Partitioning** — Non-IID partitioning of MSR-VTT by video category into 5 clients.
- [ ] **M18: Local FL Training** — FedProx + DP-SGD local optimization logic.
- [ ] **M19: Aggregation Logic** — Trimmed Mean aggregator + uniformity circuit breaker.
- [ ] **M20: DAPHW Adapter** — Domain adapter for local-only personalization.
- [ ] **M21: FL Convergence** — Achieving ≥85% of centralized R@1 over 50 rounds.
- [ ] **M22: Byzantine Defense** — Robustness verification against injected malicious clients.
- [ ] **M23: Incremental Indexing** — Fast FAISS rebuild using cached backbone features after FL updates.

### Phase 6: Ablations & Publication
- [ ] **M24: Ablation T1** — Backbone comparison (ViT vs MobileCLIP).
- [ ] **M25: Ablation T2** — Component contribution (DGSE vs THNC vs Transformer).
- [ ] **M26: Ablation T3** — FL Strategy (FedAvg vs FedProx).
- [ ] **M27: Ablation T4** — Caption density effects.
- [ ] **M28: Final Performance Table** — Zero-shot → Centralized → Federated summary.

### Phase 7: Edge Deployment (Samsung)
- [ ] **M29: Quantization** — INT8 quantization for MobileCLIP and Transformer.
- [ ] **M30: NPU Optimization** — ONNX/DLC conversion for Hexagon NPU.
- [ ] **M31: Android Integration** — FAISS JNI + SQLite/Room moment storage app.
- [ ] **M32: FL Edge Node** — Secure aggregation worker for Android background service.

---

## 🔍 How to Test Passed Milestones

### 1. Verification of M1–M3 (Infrastructure)
Run the basic sanity check to ensure the backbone and database are communicating.
```powershell
$env:PYTHONPATH = "."
python scripts/sanity_check_m8.py
```
*Expected: "Success: Model and Projection Head forward pass" and DB connection confirmed.*

### 2. Verification of M4–M7 (Baselines)
To re-run the 1K-A Zero-shot baseline (requires raw chunks and backbone):
```powershell
python scripts/eval_m10.py --checkpoint NONE
```
*Expected: R@1 = 25.8%*

### 3. Verification of M8–M10 (Training Success)
Validate the trained projection heads against the MSR-VTT 1K-A test set.
```powershell
python scripts/eval_m10.py --checkpoint checkpoints/best_model.pt
```
*Expected: R@1 = 31.2% (Improvement over baseline confirmed).*

---

## 🛠️ Installation & Setup
1. **Clone & Env**:
   ```powershell
   git clone <repo-url>
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Data**: Place `fedvcmr.db` in the root and ensure `cache/frame_features.bin` is available.
3. **Backbone**: Uses MobileCLIP-S1 (automatically downloaded on first run via `open_clip`).

---

## 📄 License & Attribution
Part of the FedVCMR Research Project. Reference papers include MA-VR (IEEE Access 2025) and MobileCLIP (CVPR 2024).
