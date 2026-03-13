# FedVCMR: Technical Reference — Chunks, Sliding Windows, FAISS & Moment Retrieval

---

## 1. Chunk Components

Each chunk is a **rich multi-modal packet** — not just a video clip, but a structured representation designed for both fast retrieval and precise temporal grounding.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CHUNK PACKET                              │
│                                                                  │
│  IDENTITY                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ chunk_id   │ video_id   │ t_start │ t_end │ scale        │   │
│  │ scene_id   │ scene_pure │ duration                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  VISUAL (stored as float16, ~8KB per chunk)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ backbone_features: (8 × 512) float16                     │   │
│  │   ← MobileCLIP-S1 vision tower output per frame          │   │
│  │   ← FROZEN, cached once, NEVER recomputed after FL       │   │
│  │                                                           │   │
│  │ frame_embeddings: (8 × 256) float32                       │   │
│  │   ← backbone_features → Vision Projection Head           │   │
│  │   ← used for MaxSim reranking (Stage 3B)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  TEXT                                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ subtitle_text: str   ← ASR/JSON overlap with [t_start,   │   │
│  │                         t_end] (TVR has this natively)    │   │
│  │ s_emb: (256,) float32 ← encoded subtitle for Transformer │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  FAISS INDEX VECTOR                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ chunk_embedding: (256,) float32, L2-normalized            │   │
│  │                                                           │   │
│  │ Center-biased weighted average of frame_embeddings:       │   │
│  │   weights = [0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5]│   │
│  │   chunk_emb = Σ(wᵢ × vᵢ) / Σwᵢ                          │   │
│  │                                                           │   │
│  │ Rationale: scene transitions cluster at chunk edges.      │   │
│  │ Center frames carry the cleanest semantic content.        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Why These Components

| Component | Purpose | Paper Backing |
|---|---|---|
| K=8 frames per chunk | Sufficient for appearance-based queries; single frame bias shows diminishing returns beyond 8 at 1fps | Lei et al. (2022) — *Single Frame Bias*, ACL 2022 |
| Per-frame storage (not just average) | Enables MaxSim reranking — cosine on averages is lossy for asymmetric text-video similarity | Khattab & Zaharia (2020) — *ColBERT*, SIGIR 2020 |
| Center-biased weighted average | Scene transitions at edges corrupt the index vector | Luo et al. (2022) — *CLIP4Clip*, Neurocomputing 2022 |
| Subtitle s_emb token | TVR shows +5–13% on dialogue-heavy content when ASR/subtitle text is fused | Lei et al. (2020) — *TVR Dataset*, ECCV 2020 |
| float16 backbone cache | Enables sub-second index rebuild after FL without re-running backbone | He et al. (2020) — *MoCo* mmap pattern |

---

## 2. Sliding Window Design

### Why Not Fixed Windows (MA-VR's Approach)

MA-VR uses fixed-stride sliding windows with no scene awareness. This produces **semantic soup** — a chunk that straddles a scene cut averages features from two completely different scenes, producing an embedding that matches neither [Luo et al., 2022 — CLIP4Clip Table 3 shows 4–8% R@1 degradation from boundary contamination].

### Our Approach: Dataset-Conditional Scene-Aware Windowing

```
MSR-VTT (pre-cut clips, 10–30s):
─────────────────────────────────────────────────────────────────
  Fixed 8s/4s windows — scene detection adds nothing
  (clips are already single-scene by dataset construction)

  [  chunk 1  ][  chunk 2  ][  chunk 3  ]
  0s    8s    4s    12s   8s    16s
       ←4s overlap→   ←4s overlap→


ActivityNet (untrimmed, 2–10 min):
─────────────────────────────────────────────────────────────────

  Step 1: TransNetV2 Scene Detection
  ════════════════════════════════════════════════════════════════
  Video: ████████████████████████████████████████████████
  Scenes:[══Scene A══════][═Scene B═][══════Scene C════════]
          0s           18.3s  22.1s                       67.4s

  TransNetV2 [Souček et al., 2020] is a lightweight CNN (~1MB)
  trained on broadcast media. Handles gradual transitions and
  dissolves — histogram differencing misses these.

  Step 2: Adaptive Multi-Scale Windowing Within Scene Boundaries
  ════════════════════════════════════════════════════════════════

  Scene A (18.3s — long):
  Scale 4s/2s stride:
  [C1:0–4s][C2:2–6s][C3:4–8s][C4:6–10s][C5:8–12s][C6:10–14s]
                                          [C7:12–16s][C8:14–18.3s]

  Scale 8s/3s stride:
  [C9:0–8s][C10:3–11s][C11:6–14s][C12:10.3–18.3s]

  Scene B (3.8s — short, below 4s min):
  ← single chunk with ±1s cross-scene context
  [C13: 21.1–23.1s] flagged scene_pure=False

  Scene C (45.3s — long):
  Scale 4s/2s + Scale 8s/3s applied within [22.1, 67.4s]
  Chunks NEVER cross the 22.1s or 67.4s boundaries.

  ════════════════════════════════════════════════════════════════
  KEY: No chunk spans two scenes. Each chunk's embedding
  represents coherent content. FAISS retrieval is clean.
```

### Window Parameters by Dataset

| Dataset | Scale A | Scale B | Min Chunk | K frames |
|---|---|---|---|---|
| MSR-VTT | 8s/4s (fixed) | — | — | 8 |
| ActivityNet | 4s/2s stride | 8s/3s stride | 3s | min(max(8, ⌈duration⌉), 32) |

### Why 2 Scales Not 3

Three scales triple the FAISS index size (~75K chunks → ~225K chunks). Memory impact on 8GB device is prohibitive. Two scales (4s + 8s) cover the p5–p95 moment length distribution in ActivityNet (~3s to ~45s) with sufficient granularity. [Krishna et al., 2017 — ActivityNet Captions, ICCV 2017 — moment length distribution analysis].

---

## 3. FAISS Mechanism

### Why FAISS Cosine on Averages Is Suboptimal

```
Query: "nurse hands over a chart"   (precise, 5 words)
Chunk: 8s of video                  (diffuse, many scenes)

Mean-pooled chunk embedding averages:
  - frame 1: empty hallway     weight 1/8
  - frame 2: nurse walking     weight 1/8
  - frame 3: NURSE + CHART     weight 1/8  ← the relevant frame
  - frame 4: doctor at desk    weight 1/8
  - frame 5: empty room        weight 1/8
  ...

Result: correct frame contributes only 12.5% to the index vector.
A wrong video with 4 mediocre frames scores higher.
```

This is the **asymmetric similarity problem**: text queries are precise; chunk embeddings are diffuse. Cosine on averages systematically penalizes videos where only a subset of frames match. [Khattab & Zaharia, 2020 — ColBERT MaxSim addresses exactly this].

### Our Two-Stage FAISS Design

```
┌─────────────────────────────────────────────────────────────────┐
│  INDEX A — Chunk-Level (Coarse Retrieval)                        │
│                                                                  │
│  Vectors:    chunk_embedding (256-d, L2-normalized)              │
│  Index type: IVFFlat, nlist=256                                  │
│              ↑ NOT IVF-PQ — PQ breaks incremental add()         │
│               after FL rounds [Johnson et al., 2021 — FAISS]     │
│  Size:       ~5MB for 10K chunks                                 │
│  Query time: ~10ms                                               │
│  Returns:    top-100 chunk_ids                                    │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3B — MaxSim Frame Reranking (DGSE)                        │
│                                                                  │
│  For each of top-100 chunks × 8 frames × 3 query phrasings:     │
│                                                                  │
│    fine_score(Q, Cⱼ) = (1/8) Σᵢ max_k cosine(q_emb_k, vᵢ)     │
│                                                                  │
│    ↑ Each frame finds its BEST matching query phrasing           │
│    ↑ A single matching frame rescues the whole chunk             │
│                                                                  │
│  Cost: 100 chunks × 8 frames × 3 phrasings = 2,400 dot products │
│  At 256-d: ~15ms on ARM CPU                                      │
│  Result: top-10 chunks for Cross-Modal Transformer               │
└─────────────────────────────────────────────────────────────────┘
```

### Why IVFFlat Not IVF-PQ

PQ (Product Quantization) compresses vectors by training a codebook on the corpus. Adding new vectors after FL updates requires either retraining the codebook (expensive) or accepting quality degradation. **IVFFlat supports `index.add()` incrementally** — new videos index in O(N) without rebuilding. This is non-negotiable for the FL post-round re-indexing requirement. [Johnson et al., 2021 — FAISS, IEEE].

---

## 4. Moment Retrieval Mechanism vs. MA-VR

### MA-VR's Approach (What We're Beating)

```
MA-VR Pipeline:
  1. Average all frame embeddings → single video vector
  2. φ^MA filter generated from POOLED (averaged) features
  3. Single-layer cross-modal encoder
  4. Boundary prediction via 1D Conv over uniform feature sequence
  5. S_VCMR = S_MR × S_VR

Core weaknesses:
  ✗ φ^MA generated from averaged features — temporal order lost BEFORE filter
  ✗ Single-layer encoder too shallow for fine-grained temporal alignment
  ✗ No scene awareness — chunks cross semantic boundaries
  ✗ Fixed-stride windows — same granularity for 5s and 5min moments
  ✗ No asymmetric retrieval handling (cosine on averages)
```

### Our Approach (Stage 4 Cross-Modal Transformer)

```
Input per top-10 chunk:
  V = [v₀(256), v₁(256), ..., v₇(256)]  ← per-frame, WITH positional encoding
  Q = query embedding (256-d)
  s_emb = subtitle embedding (256-d)

Step A — Query-Conditioned Frame Attention:
  αᵢ = softmax(vᵢ · Q / √256)           ← frame-query affinity
  V_q = Σᵢ αᵢ × vᵢ                      ← query-attended video summary

Step B — 2-Layer Cross-Modal Transformer:
  input_seq = [Q_proj; v₀; v₁; ...; v₇; s_emb]   ← 10 tokens
  V_cross = TransformerEncoder(input_seq, layers=2, heads=4, dim=256)
  [_, v*₀, ..., v*₇, _] = V_cross
  ↑ v*ᵢ = frame i in context of query AND neighboring frames

Step C — Motion-Aware Temporal Filter (Novel vs MA-VR):
  frame_relevance_i = cosine(v*ᵢ, Q)
  φ^MA = GaussianSmooth(frame_relevance, σ=1.5)  ∈ R^8
  ↑ Filter generated from per-frame CROSS-MODAL features, not pooled features
  ↑ This is the core fix for MA-VR's main weakness

Step D — Boundary Prediction:
  τ_s, τ_e = Conv1D([v*₀,...,v*₇] × φ^MA)

Step E — Relevance Score:
  r_VR = MLP([V_q ; WeightedPool(V_cross × φ^MA) ; s_emb ; max(frame_relevance)])

Final VCMR Score:
  S_VCMR = S_MR × S_VR × S_MaxSim^0.5
```

### Side-by-Side Comparison

| Dimension | MA-VR | FedVCMR | Advantage |
|---|---|---|---|
| Frame representation | Averaged → single vector | Per-frame sequence + positional encoding | Temporal order preserved |
| φ^MA filter source | Pooled (averaged) features | Per-frame cross-modal attended features | Query-conditional at frame level |
| Cross-modal encoder | 1-layer Transformer | 2-layer Transformer | Greater representational depth |
| Segmentation | Fixed stride | Scene-aware adaptive multi-scale | No cross-scene contamination |
| Retrieval | FAISS cosine on averages | FAISS coarse → MaxSim frame reranking | Handles asymmetric similarity |
| Subtitle fusion | Not used | s_emb as 10th Transformer token | +5–13% on subtitle-rich content |
| Score combination | S_MR × S_VR | S_MR × S_VR × S_MaxSim^0.5 | MaxSim boost for precise matches |

**Paper backing for 2-layer vs 1-layer:** UniVTG [Lin et al., ICCV 2023] shows per-frame cross-attention achieves within 0.18 mAP of full cross-attention — validating our lightweight 2-layer approximation as near-optimal. QD-DETR [Moon et al., CVPR 2023] demonstrates query-dependent frame weighting improves moment retrieval by 4–6 mAP over query-independent approaches.

---

## 5. VLM Comparison and Final Choice

### Candidate VLMs Considered

| VLM | Params | Type | Edge-Viable? | Zero-Shot Retrieval | Notes |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 151M | Contrastive | ⚠️ Slow | 30.4% R@1 MSR-VTT | Baseline; MA-VR uses this |
| CLIP ViT-L/14 | 428M | Contrastive | ❌ No | ~37% R@1 | Too large for edge |
| **SigLIP-SO400M** | 400M | Contrastive (sigmoid) | ❌ No | Best zero-shot | Server teacher only |
| **SigLIP-B/16** | 86M | Contrastive (sigmoid) | ⚠️ Flagship only | ~32% R@1 | Too heavy for A55 |
| MobileCLIP-S0 | 11M | Contrastive (distilled) | ✅ Yes | ~25–28% R@1 | Too small — leaves accuracy on table |
| **MobileCLIP-S1** | 30M | Contrastive (distilled) | ✅ Yes | ~29–32% R@1 | **CHOSEN** |
| MobileCLIP-S2 | 35M | Contrastive (distilled) | ✅ Partial | ~31–34% R@1 | No stable public checkpoint |
| Florence2-base | 230M | Generative | ⚠️ Moderate | Not retrieval-native | Good captioning, not indexing |
| InternVL2-8B | 8B | Generative | ❌ Server only | N/A | Caption augmentation server-side |
| InternVideo2-1B | 1B | Video-Language | ❌ Server only | N/A | Temporal distillation teacher |

### Why Not SigLIP-SO400M as Backbone

SigLIP uses sigmoid loss instead of softmax — this improves zero-shot retrieval by removing the dependence on in-batch negatives during pretraining [Zhai et al., 2023 — SigLIP, ICCV 2023]. SO400M achieves state-of-the-art zero-shot performance.

**However:** 400M params = ~400MB FP32 / ~100MB INT8. Even at INT8 this **does not fit in Hexagon NPU SRAM** (Samsung A55 has ~40MB SRAM available for inference). This causes DDR memory bandwidth bottlenecks that multiply inference latency by 3–5×. The model is relegated to server-side teacher use only.

SigLIP-B/16 at 86M params / ~22MB INT8 is borderline — fits S24 flagship but fails on A55 (6GB RAM, weaker NPU). Rejected for the same reason.

### Why MobileCLIP-S1 Is the Correct Choice

```
MobileCLIP-S1 advantages for Samsung NPU:

1. 30MB INT8 fits ENTIRELY in Hexagon NPU SRAM
   → Zero DDR memory bandwidth bottleneck
   → 12ms vision encoding (vs 210ms for SigLIP-B/16)

2. Multi-teacher distillation from DataCompDR-1B
   → Retains 95%+ of CLIP ViT-B/32 retrieval quality
   → At 20% the parameters [Vasu et al., CVPR 2024]

3. Hybrid CNN-Transformer architecture
   → Depthwise separable convolutions map to Hexagon DSP
   → ViT layers map to Hexagon NPU
   → Optimal hardware utilization split

4. Distilled from CLIP family
   → Same embedding geometry as CLIP
   → Compatible with CLIP-trained contrastive objectives
   → No cross-architecture alignment problem
      (unlike SigLIP→MobileCLIP which would require CRD)

5. Both towers output 512-d
   → Projected to 256-d via trainable heads
   → 256-d fits L2 cache; 4× less DDR than 512-d
```

**S1 vs S0 decision:** S0 (11M params) was the original HLD choice under "build what you ship" reasoning. However S1 at 30MB INT8 still fits NPU SRAM — the key property — while providing ~4–6% higher zero-shot R@1 on retrieval benchmarks. When competing against MA-VR's 10.04% R@1 IoU=0.7, every percentage point matters. [Sun et al., ACL 2023 — EfficientVLM: "at 25% original FLOPs, R@1 drops only 2.1 points"].

### Server-Side VLMs (Never Deployed on Device)

| VLM | Role | When Used |
|---|---|---|
| SigLIP-SO400M | Temporal distillation teacher | Server training only — generates soft labels for segment-level temporal alignment |
| InternVL2-8B | Caption augmentation | Offline dataset prep — generates 5 diverse captions per keyframe for richer training pairs |
| InternVideo2-1B | Video temporal teacher | Knowledge distillation — teaches MobileCLIP-S1 temporal consistency from full video clips |

---

## Reference Papers

| Paper | Venue | Relevance |
|---|---|---|
| Choi et al. — MA-VR | IEEE Access 2025 | System we beat; source of curriculum negatives and joint VR+MR |
| Vasu et al. — MobileCLIP | CVPR 2024 | Our backbone; multi-teacher distillation validation |
| Zhai et al. — SigLIP | ICCV 2023 | Sigmoid loss advantage; rejected for edge size |
| Radford et al. — CLIP | ICML 2021 | Foundation embedding space |
| Khattab & Zaharia — ColBERT | SIGIR 2020 | MaxSim late interaction motivates Stage 3B |
| Johnson et al. — FAISS | IEEE TBD 2021 | IVFFlat incremental add() requirement |
| Moon et al. — QD-DETR | CVPR 2023 | Query-dependent frame weighting (+4–6 mAP) |
| Lin et al. — UniVTG | ICCV 2023 | Per-second saliency near-optimal (0.18 mAP gap vs cross-attention) |
| Lei et al. — Single Frame Bias | ACL 2022 | K=8 frames sufficient; diminishing returns beyond |
| Souček et al. — TransNetV2 | arXiv 2020 | Scene detection handles gradual transitions |
| Krishna et al. — ActivityNet | ICCV 2017 | Primary eval benchmark; moment length distribution |
| Luo et al. — CLIP4Clip | Neurocomputing 2022 | Mean pooling degrades 4–8% R@1 on longer videos |
| Sun et al. — EfficientVLM | ACL 2023 | Aggressive compression retains 97.5% retrieval quality |
| Lei et al. — TVR | ECCV 2020 | Subtitle fusion +5–13% on dialogue-heavy content |

---

## 6. Training Pipeline Refinements (M9/M10 Breakthroughs)

During the intensive training and validation phase (March 2026), the pipeline was significantly stabilized to move from a 25.8% (Zero-shot) baseline to a **31.2% (+5.4%)** retrieval score on the official 1K-A test set.

### 6.1 The Data Alignment "Silent Killer"

**The Bug:** The binary visual cache (`frame_features.bin`) was built using `ORDER BY chunk_id` (String-sort: video0, video1000, video1001...). However, some early training scripts used `ORDER BY rowid` (Numeric-sort: video0, video1, video2...). This caused the model to train on effectively **random noise** (pairing text with the wrong visual chunks).

**The Fix:** All training and evaluation queries are now strictly synchronized to **`ORDER BY chunk_id`**. This restored the visual-semantic alignment necessary for the model to converge.

### 6.2 Architecture for Generalization

Initially, complex MLP heads were used, which led to **dimensional collapse** (temperature falling to 0.031) and extreme overfitting (heads producing orthogonal outputs for unseen data).

**Current Stabilized Architecture:**
- **Identity-Linear Head**: Single linear layer (512→512) initialized as an Identity matrix.
- **Rationale**: CLIP already has world-class alignment. The projection head's job is to learn **small task-specific refinements**, not to reinvent the embedding space from scratch.
- **Hyperparameters**: Fixed Temperature (**0.07**), Weight Decay (**0.1**), and LR (**3e-4**) with 2-epoch warmup.

### 6.3 Metric Progress

| Milestone | Strategy | R@1 (1K-A) | Outcome |
|---|---|---|---|
| M3/M4 | Zero-Shot MobileCLIP-S1 | 25.8% | Baseline |
| M9 (v1/v2) | Learnable Temp + MLP Head | 4.4% | FAILED (Overfit Train Split) |
| **M10 (v4)** | **Fixed Alignment + Identity Linear** | **31.2%** | **PASSED** |
