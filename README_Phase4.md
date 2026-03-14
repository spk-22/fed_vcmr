# Phase 4 Results: Cross-Modal Transformer (M13-M16)

## Overview
Phase 4 focused on transitioning from simple vector-based retrieval to a **Cross-Modal Transformer** grounding head. This milestone successfully implemented a fine-grained temporal localizer that adjusts video segment boundaries based on specific query semantics.

## Technical Implementation
- **Architecture**: A 2-layer, 4-head Transformer encoder (256-dim) implemented in `src/transformer_model.py`.
- **Hybrid Reranking Flow**:
    1. **Selection (MaxSim)**: Identifies the Top-K most relevant chunks using the frame-level matching logic established in Phase 3.
    2. **Refinement (Transformer)**: Predicts normalized [start, end] adjusters for each selected chunk.
    3. **Window Merging**: Merges adjacent high-confidence chunks from the same video into a single unified temporal window to handle longer ground-truth segments.

## Numerical Results
Evaluated on the **ActivityNet Validation Subset** using the `transformer_best.pt` model trained on real MobileCLIP features.

| Metric | Phase 4 Hybrid Result | Phase 4 Initial (Coarse) | Phase 2 Zero-Shot |
| :--- | :--- | :--- | :--- |
| **Recall@1 (IoU=0.7)** | **7.23%** | 0.00% | 0.00% |
| **Recall@1 (IoU=0.5)** | **11.45%** | 2.67% | 10.53% |
| **Recall@1 (IoU=0.3)** | **21.69%** | 4.00% | 21.05% |

### Performance Analysis
- **Grounding Gain**: The introduction of the Transformer allowed us to hit **7.23%** at the strict **IoU=0.7** threshold where previously we achieved 0%. 
- **IoU=0.5 Stability**: We maintained/improved our baseline performance while narrowing down the segment precision.
- **Scaling Note**: These results were achieved on a subsampled training set (1K pairs). To reach the state-of-the-art target (>10.04%), training on the full 130K ActivityNet pairs is required.

## Reproducing Results
1. **Training**: Use `Phase4_Transformer_Training_V2.ipynb` in Google Colab with the `colab_export` data packet.
2. **Inference**: Place the resulting `transformer_best.pt` in the `cache/` directory.
3. **Execution**: Run `python eval_activitynet.py`.

---
*Results verified on 2026-03-14*
