# Milestone 6: ActivityNet Data & Evaluation Pipeline

## Objective
The goal of **Milestone 6 (M6)** was to prepare the system for processing the secondary dataset (ActivityNet) and to build out the evaluation metrics specifically required for moment retrieval.

### Requirements:
- Parse ActivityNet annotations.
- Download ~200 videos to serve as a test sample.
- Have the evaluation pipeline ready.
- **Gate**: The Intersection over Union (IoU) metric must compute correctly.

---

## Results & Achievements

### 1. Data Parsing & Download
Instead of parsing the massive raw ActivityNet dataset, we utilized a curated subset of 200 samples hosted on HuggingFace (`activitynet_200_samples.json`).
- We successfully wrote a robust ingestion script (`src/download_activitynet.py`) that parses this JSON format.
- Using `yt-dlp` under the hood, the script automatically downloaded all reachable YouTube videos from this subset directly into `data/activitynet/videos/`. 
- **Result:** Successfully downloaded **155 videos** in `.mp4` format (the remaining ~45 videos were skipped as they were either marked private or deleted by YouTube). This is a sufficient sample size for prototyping and evaluating Milestone 7.

### 2. Evaluation Pipeline (IoU Metrics)
We extended our evaluation engine (`src/evaluation.py`) to handle temporal segments alongside standard vector retrieval ranks.
- **Temporal IoU (`compute_iou`)**: Implemented mathematically sound overlap calculation between two temporal moments (Start Time → End Time).
- **Moment Retrieval Evaluation (`evaluate_moment_retrieval`)**: Added logic to compute Recall@1 and Recall@5 at specified IoU thresholds (e.g., IoU=0.5, IoU=0.7). 

### 3. Verification (Gate Passed)
To ensure the mathematical correctness of our core gate requirement, we implemented a dedicated unit test suite (`tests/test_metrics.py`).
- The suite tests exact matches, non-overlapping segments, partial overlaps, and encapsulated overlaps.
- **Result:** All tests passed successfully. The IoU metric computes correctly and accurately identifies R@1 threshold breaks.

---

## Conclusion
**Milestone 6 is complete.** We achieved everything we set out to do for this phase. 
We now have the ActivityNet data stored locally and a verified metric pipeline, bringing us perfectly in position to start **Milestone 7** (Zero-shot moment retrieval on the ActivityNet val subset).
