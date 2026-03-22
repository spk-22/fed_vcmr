# FedVCMR — Project Milestones

## Phase 1: Working Retrieval (Base Infrastructure)
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M1 | Core VLM Setup | MobileCLIP-S1 integration, 512-d image+text encoding | ✅ Done |
| M2 | Chunking Engine | Frame extractor + sliding window (8s/4s stride) | ✅ Done |
| M3 | Minimal Search | SQLite metadata + basic FAISS query pipeline | ✅ Done |

## Phase 2: Real Data & Baselines
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M4 | Mass Ingestion | MSR-VTT 10k indexed (6,513 videos), binary memmap cache | ✅ Done |
| M5 | Zero-Shot Baseline | MSR-VTT 1K-A eval → **R@1 = 25.8%** (target >20%) | ✅ Done |
| M6 | ActivityNet Support | Annotations parsed, 3,302 videos downloaded | ✅ Done |
| M7 | Moment Evaluation | Zero-shot moment retrieval on ActivityNet val subset | ✅ Done |

## Phase 3: Centralized Training
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M8 | Projection Learning | Linear projection heads (512→512) + InfoNCE loss | ✅ Done |
| M9 | Dataset Alignment | Fixed chunk_id sort order mismatch between cache and training | ✅ Done |
| M10 | Centralized Gate | MSR-VTT R@1 = **31.2%** (+5.4% over zero-shot) | ✅ Done |
| M11 | DGSE Integration | Dual-granularity pooling (coarse + fine MaxSim) | ✅ Done |
| M12 | Moment Loss | Joint MR + VR + Temporal boundary loss optimization | ✅ Done |

## Phase 4: Cross-Modal Transformer & Grounding
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M13 | Transformer Arch | 2-layer, 4-head, 256-dim Cross-Modal Transformer. **R@1 IoU@0.5 = 34.5%** (3.4x MA-VR) | ✅ Done |

## Phase 5: Pipeline Integration & Scale-Up Evaluation
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M14 | Pipeline Integration | Chain FAISS→MaxSim→DGSE→Transformer. Output: `[vid, start, end, conf]` | ✅ Done |
| M15 | Latency Profiling | Per-stage latency measurement. Target: <100ms total (Achieved: **42.9ms**) | ✅ Done |
| M16 | ActivityNet Full Eval | R@1 IoU@0.5: **6.47%**, IoU@0.7: **3.33%** (vs 10.04% baseline) | ✅ Done |
| M17 | Charades-STA Eval | R@1 IoU@0.5: **1.90%**, IoU@0.7: **0.92%**, Video R@1: **5.07%** (1,736 videos, zero-shot) | ✅ Done |
| M18 | Kinetics-400 Eval | Extract features from subset, build FAISS index, R@1/5/10 domain generalization (eval only, no GPU) | ⬜ |

## Phase 6: Robustness & Refinement
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M19 | Query Robustness | Descriptive R@1: **32.47%** vs Ambiguous R@1: **15.91%** — Gap: **+16.56%** | ✅ Done |
| M20 | Synonym Substitution | Swap keywords with synonyms. Measure R@1 and IoU@0.5 stability (no GPU) | ⬜ |
| M21 | THNC Training | 3-phase hard negative curriculum (Easy→Medium→Hard). Expected: +2-3% R@1 (Colab T4) | ⬜ |
| M22 | ActivityNet Zero-Shot | Zero-shot baseline on ActivityNet to contextualize transformer improvement | ⬜ |

## Phase 7: Model Export & Quantization
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M23 | DAPHW | Domain-Adaptive Projection Head Warm-Start. Local adapter per domain, never federated | ⬜ |
| M24 | ONNX Export | Export projection heads + transformer to ONNX. Verify outputs match PyTorch ±1e-4 | ⬜ |
| M25 | TFLite Conversion | Convert ONNX→TFLite. INT8 quantization for mobile | ⬜ |

## Phase 8: Federated Learning via Android Studio
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M26 | Android Client Setup | Create Android Studio project with TFLite SDK. Each emulator instance = 1 FL client | ⬜ |
| M27 | On-Device Training | Implement local fine-tuning on each Android client using TFLite + on-device data partition | ⬜ |
| M28 | FL Simulation (5 Clients) | Run 5 Android emulators as non-IID clients (cooking/sports/travel/music/mixed, ~650 vids each). 50 FL rounds, FedProx μ=0.01, Trimmed Mean (10%), DP-SGD ε≤10 | ⬜ |
| M29 | FL Ablation | Compare FedAvg vs FedProx vs Trimmed Mean. Report R@1 + communication cost per round | ⬜ |

## Phase 9: Ablation & Reporting
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M30 | Full Ablation Table | Remove one component at a time (No DGSE / No Transformer / No FL / No THNC). Measure deltas | ⬜ |
| M31 | Methodology Report | Full writeup: 3 novel contributions, architecture, all benchmarks M5–M30, comparison vs MA-VR | ⬜ |
| M32 | Paper Draft | Abstract, Intro, Related Work, Methodology, Experiments, Ablation, Conclusion | ⬜ |

## Phase 10: Deployment on Physical Device
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M33 | Galaxy S24 Benchmark | Deploy TFLite model on real Galaxy S24. Per-stage latency on Hexagon NPU. Target: <100ms | ⬜ |
| M34 | Galaxy A55 Benchmark | Minimum-spec device test. Establishes minimum hardware requirement for paper | ⬜ |
| M35 | Android App | Full search app: text query → video result with temporal highlight. FAISS JNI + SQLite/Room | ⬜ |
| M36 | FL Edge Node | Secure aggregation worker as Android background service for production FL | ⬜ |

---

## Summary

| Phase | Milestones | Status |
|-------|-----------|--------|
| Phase 1: Infrastructure | M1–M3 | ✅ Complete |
| Phase 2: Baselines | M4–M7 | ✅ Complete |
| Phase 3: Training | M8–M12 | ✅ Complete |
| Phase 4: Transformer | M13 | ✅ Complete |
| Phase 5: Integration & Eval | M14–M18 | ⬜ Next |
| Phase 6: Robustness | M19–M22 | ⬜ Planned |
| Phase 7: Export | M23–M25 | ⬜ Planned |
| Phase 8: FL (Android Studio) | M26–M29 | ⬜ Planned |
| Phase 9: Reporting | M30–M32 | ⬜ Planned |
| Phase 10: Deployment | M33–M36 | ⬜ Final |

**Key Results So Far:**
- MSR-VTT R@1: **31.2%** (M10)
- ActivityNet R@1 IoU@0.5: **34.5%** — beats MA-VR by **3.4x** (M13)
- Zero-Shot Generalization: Verified on external YouTube video
- Inference Latency: **~63ms** end-to-end on consumer hardware
