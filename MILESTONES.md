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
| M20 | VQRF v2 Implementation | Three-path routing (Rule/Gated/Pass) + Object Veto Classifier. | ✅ Done |
| M21 | VQRF v2 Results | **35.77% R@1** (Descriptive), **30.70%** (Overall). Zero regression success. | ✅ Done |
| M22 | Honest Audit | Verified mathematical ceiling: MaxSim (28.3%) / MCR (26.6%) vs 31.7% base. | ✅ Done |

## Phase 7: Federated Learning Simulation (Flower)
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M23 | Non-IID Sharding | Split 7k train set into 4 categorical shards (Gaming/Sports/Cooking/News). | ✅ Done |
| M24 | FL Simulation (4 Clients)| 50 rounds of FedAvg + Trimmed Mean (10%) aggregation using Flower. | 🔄 Next |
| M25 | DAPHW Verification | Measure Personalization Gain (+X% R@1) using Hybrid Weighting (Global + Local).| ⬜ |

## Phase 8: Edge Device Deployment (Android Studio)
| # | Milestone | Description | Status |
|---|-----------|-------------|--------|
| M26 | TFLite/ONNX Export | Export 512-d text/vision heads + Transformer to quantized INT8 TFLite. | ⬜ |
| M27 | Kotlin Adaptation | Port VQRF v2 & DGSE pooling logic to Android (Kotlin/Room). | ⬜ |
| M28 | Edge Latency Audit | Benchmark on Galaxy S24/A55 Hexagon NPU. Target: <100ms. | ⬜ |
| M29 | Real-Device FL Node | Connect physical device to Flower server as 5th client node. | ⬜ |

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
