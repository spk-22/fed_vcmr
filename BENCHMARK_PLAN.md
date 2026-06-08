# On-Device VCMR Benchmark Plan (24-Hour / 10,000 Query)

This document outlines the strategy to scale the Video Corpus Moment Retrieval (VCMR) system to handle 24 hours of video content and perform 10,000 ground-truth query evaluations natively on an Android device.

> **Query strategy — 24-hour corpus + NLP-augmented 10k queries:**
> `val_1.json` has 17,505 sentence pairs across 4,917 videos (~161 hours). A 24-hour corpus uses only **733 videos → 2,638 ground-truth queries**. To reach 10,000 queries while keeping all ground truth valid, the 2,638 base sentences are augmented PC-side into ~4 variants each using NLP reframing. All 10k queries retain the same `(video_id, t_start, t_end)` ground truth. The phone does no augmentation — it receives pre-generated query strings.

---

## DO NOT HARM THE WORKING APP — Read This First

The Android app on the phone is currently **fully working** with the Charades-STA payload. Every step in this plan that touches `/sdcard/fedvcmr/` on the phone risks breaking it if done incorrectly. Follow these rules without exception before touching the phone.

### Rule 1 — Back up the phone payload before any adb push
```bash
adb pull /sdcard/fedvcmr/ ./phone_backup_charades/
```
Do this once before any other adb command in this plan. If anything goes wrong, restore with:
```bash
adb push ./phone_backup_charades/fedvcmr /sdcard/
```

### Rule 2 — Never push a file in the wrong format
The three files the app reads have strict schemas. Pushing the wrong format causes silent failures (zero results, crashes, or garbage scores) with no obvious error message.

| File on phone | Required format | What NOT to push |
|---|---|---|
| `user_index.json` | JSON **array** of objects with keys: `video_id`, `duration`, `num_frames`, `file_hash`, `ingested_at` | `phone_payload/index.json` — this is Charades-STA format (`chunk_id`, `start`, `end`) and is **incompatible** |
| `user_features_index.json` | JSON **object** keyed by `video_id` → byte offset integer | Any index keyed by `chunk_id` (e.g. `"REWLB_c0"`) — keys must match `video_id` exactly |
| `user_features_blob.bin` | Raw IEEE 754 float32, little-endian, `num_frames × 512 × 4` bytes per video | A blob extracted with 16 frames when the index says `"num_frames": 8` — sizes won't align |

### Rule 3 — Always write `num_frames` explicitly in the index
The Android app defaults to `num_frames = 16` if the field is absent. The Python extractor uses **8 frames**. A missing `num_frames` field causes the app to read twice as many bytes per video from the blob, producing completely wrong embeddings and scores. Every entry in `user_index.json` must contain `"num_frames": 8`.

### Rule 4 — Verify the new payload on PC before pushing to the phone
Run `scripts/phase0_validation.py` (or equivalent) against the generated files on PC first. Confirm:
- Entry count in `user_index.json` matches entry count in `user_features_index.json`.
- Byte offset of last entry + `(num_frames × 512 × 4)` equals the file size of `user_features_blob.bin`.
- All `video_id` values in `user_index.json` have a corresponding key in `user_features_index.json`.

### Rule 5 — Do not modify assets bundled inside the APK
The following are baked into the APK and must not be changed during this benchmark:
- `assets/models/mobileclip_text_encoder.ptl`
- `assets/models/text_head_best_model.tflite`
- `assets/models/mobileclip_vision_encoder.ptl`
- `assets/models/vision_head_best_model.tflite`

Replacing these requires a full APK rebuild and reinstall. The benchmark uses only the sdcard payload files.

### Rule 6 — Do not implement C4 (LruCache) or C2 (dot product matrix) without a test build first
`SearchEngine.java` is live in the working app. Any code change requires rebuilding and reinstalling the APK. Do not push `.java` edits without a passing local build (`./gradlew assembleDebug`).

---

## Part 0: Efficient Data Acquisition (24 Hours of Video)

To get 24 hours of video, we use the **ActivityNet Captions** dataset.
Average ActivityNet video length is **~118 seconds (~2 minutes)**.
*   **Exact target:** **733 videos = 24.02 hours** (86,466 seconds when sorted by video_id from `val_1.json`).
*   **Safety Margin:** Download **800 videos** to account for corrupted files or failed links, then trim to the first 733 that yield 24h.
*   **Ground-truth queries from these 733 videos:** **2,638 sentence-timestamp pairs** — this is the accuracy evaluation set.

> **IMPORTANT — Use the validation split, not training:** `scripts/download_anet.py` currently targets `ActivityNet/train.json` (3,300 training videos). Edit it to read from `ActivityNet/val_ids.json` and download only the first 800 IDs. The 733-video corpus must be the same videos referenced in `val_1.json` for ground truth to be valid.

### The "Fast-Download" Strategy:
1.  **Low-Resolution Encoding:** Do not download 1080p or 4K. Use `yt-dlp` to force **360p** (`-f "best[height<=360]"`). This reduces the 24-hour dataset size from ~50GB to **~3GB**, making it possible to store on a standard phone.
2.  **Multi-threaded Downloader:** Use the existing `scripts/download_anet.py` which employs `ThreadPoolExecutor` (4 workers by default).
3.  **Command:**
    ```bash
    # Edit scripts/download_anet.py to read from ActivityNet/val_ids.json
    # and set MAX_VIDS to 1000
    python scripts/download_anet.py
    ```
4.  **Hardware Acceleration:** If your PC has an NVIDIA GPU, add `--downloader ffmpeg --downloader-args "ffmpeg:-hwaccel auto"` to the `yt-dlp` command inside the script for faster merging.

---

## Part 0.5: PC-Side Query Augmentation (2,638 → 10,000)

**Goal:** Expand the 2,638 ground-truth sentences from the 733-video corpus into 10,000 NLP-reframed variants. Every variant inherits the same `(video_id, t_start, t_end)` ground truth, so Recall@K is fully meaningful for all 10k queries.

**Why PC-side only:** The phone has no augmentation capability. `TextEncoder.java` is a pure tokenize→embed pipeline (CLIP BPE, 77-token max). `SearchEngine.java` already applies 6 prompt templates internally per search call — those are separate from the query strings sent to it.

### The 4 Augmentation Passes (~4 variants × 2,638 = ~10,552, trimmed to 10,000):

**Pass 1 — Baseline (2,638 queries)**
Raw sentences from `val_1.json` as-is. No transformation.

**Pass 2 — Prefix/Subject Swap (uses existing `EXPANSIONS` dict from `scripts/ablations/eval_mcr_vqrf.py`)**
Apply the 4 synonym rules already in the codebase:
```python
EXPANSIONS = {
    'someone ':  ['a person ', 'a human '],
    'a person ': ['someone ', 'a human '],
    'people ':   ['a group ', 'individuals '],
    'a man ':    ['a guy ', 'a person '],
}
```
For each base query matching a prefix, emit one swapped variant. Yields ~800–1,200 new queries depending on prefix coverage.

**Pass 3 — Ambiguity Injection (spaCy POS tagging)**
Strip adjectives (ADJ tokens) and named entities from each sentence, keeping only verbs, nouns, and prepositions. Tests robustness to vague/underspecified queries:
- `"a man in a red jacket climbing a steep rocky mountain"` → `"a man climbing a mountain"`
- `"two women dancing near a fountain in a park"` → `"women dancing near a fountain"`

Requires: `pip install spacy && python -m spacy download en_core_web_sm`

**Pass 4 — Template Wrap**
Prepend `"a video of"` or `"a video showing"` to the core phrase (mirrors 2 of the 6 prompt templates already in `SearchEngine.java`). Tests whether surface framing affects retrieval vs the phone's internal ensembling:
- `"someone cooking pasta"` → `"a video of someone cooking pasta"`

### Output format (`benchmark_queries_10k.json`):
```json
[
  {
    "query": "a man climbs a rocky wall outdoors",
    "video_id": "v_xxxxxxx",
    "t_start": 12.4,
    "t_end": 45.1,
    "augmentation": "baseline",
    "base_query": "a man climbs a rocky wall outdoors"
  },
  {
    "query": "a person climbs a rocky wall outdoors",
    "video_id": "v_xxxxxxx",
    "t_start": 12.4,
    "t_end": 45.1,
    "augmentation": "prefix_swap",
    "base_query": "a man climbs a rocky wall outdoors"
  }
]
```

The `augmentation` tag lets Phase 4 break down Recall@K **per augmentation type** — revealing whether the phone handles ambiguous or reframed queries differently from baseline.

### DGSE upgrade path (future):
`dgse.tflite` is bundled in the APK assets (`528 KB`) but **not wired into `SearchEngine.java`**. Once integrated, it would act as a reranker specifically for Pass 3 (ambiguous queries), which are the hardest set. Wire it in via `SearchService.java` Stage 2 stub when ready.

---

## Phase 1: PC-Side Feature Extraction & Packaging

**Goal:** Avoid the battery-intensive "Ingestion" phase on the phone.

1.  **Feature Extraction:** Run `scripts/extract_anet_val_features.py`. This uses your PC's GPU to process the 1,000 videos through the MobileCLIP-S1 Vision Backbone.
2.  **Binary Packaging:** The extractor samples **8 frames per video** (not 16).
    *   1,000 videos × **8 frames** × 512 dimensions = **4,096,000 floats**.
    *   **File:** `features_blob.bin` (**~16 MB**, not 32 MB).
    *   **Index:** `features_index.json` (maps chunk IDs to byte offsets — note the actual filename produced by `scripts/pack_features_blob.py` is `features_index.json`, not `user_index.json`).
3.  **Generate a properly-formatted `user_index.json` for ActivityNet:**
    The existing `phone_payload/index.json` is in Charades-STA format (`chunk_id`, `start`, `end`) and is **not compatible** with what `SearchEngine.java` expects. You must generate a new index with the correct keys before pushing. `extract_anet_val_features.py` or `pack_features_blob.py` should emit each entry as:
    ```json
    {
      "video_id": "v_xxxxxxx",
      "duration": 123.4,
      "num_frames": 8,
      "file_hash": "<md5 of first 1MB>",
      "ingested_at": 0
    }
    ```
    Similarly, `user_features_index.json` must key by `video_id` (not `chunk_id`).

4.  **Transfer (The "Magic" Step):**
    > **WARNING:** This overwrites the current working Charades-STA payload on the phone. Back it up first if needed:
    > ```bash
    > adb pull /sdcard/fedvcmr /path/to/backup/
    > ```
    ```bash
    adb push phone_payload/features_blob.bin      /sdcard/fedvcmr/user_features_blob.bin
    adb push phone_payload/features_index.json    /sdcard/fedvcmr/user_features_index.json
    adb push phone_payload/anet_user_index.json   /sdcard/fedvcmr/user_index.json
    ```
    *Now the phone instantly "has" 24 hours of video indexed without doing any work.*

---

## Phase 2: Search Engine Latency Optimizations

`SearchEngine.java` already has several optimizations implemented. Status per item:

1.  **A1: Prompt Ensembling (Accuracy) — ALREADY IMPLEMENTED:**
    *   6 templates are already active: `"%s"`, `"a video of %s"`, `"a photo of %s"`, `"someone %s"`, `"a person %s"`, `"a video showing %s"`.
    *   Embeddings are averaged into a single ensemble vector per query.
    *   No code change needed.

2.  **A2: Hubness Suppression — ALREADY IMPLEMENTED:**
    *   Inverted Softmax penalty applied as `finalScore = smoothedScore - (HUB_LAMBDA * (v.meanSim - globalMean))` with `HUB_LAMBDA = 0.35`.

3.  **A3: Temporal Smoothing — ALREADY IMPLEMENTED:**
    *   3-frame window averaging across adjacent frames.

4.  **C2: Precompute Dot Products (Speed) — PARTIALLY IMPLEMENTED:**
    *   Frames are cached in RAM via `VideoData` objects, and dot products are computed inline during linear scan.
    *   A fully precomputed `allScores[video][frame]` 2D matrix is **not yet built**. This is a remaining optimization opportunity: pre-scan all ~8,000 frames once per query and store results before applying smoothing/hubness passes.

5.  **C4: Query Embedding Cache — NOT YET IMPLEMENTED:**
    *   Add `LruCache<String, float[]>` to cache the 512-dim text embedding per query string.
    *   This drops latency from ~350ms to ~1ms for repeated queries (important for the benchmark loop which may repeat queries across runs).
    *   Implement in `SearchEngine.java` before running Phase 3.

---

## Phase 3: Benchmark Execution Framework

> **Prerequisite:** `BenchmarkService.java` currently only stubs intent logging — the actual benchmark loop is **not yet implemented**. Complete it before this phase.

1.  **Memory Mapping (MMAP):**
    *   Use `MappedByteBuffer` to read `features_blob.bin`.
    *   Each video chunk is `8 × 512 × 4 = 16,384 bytes`.
    *   This keeps RAM usage at **~40MB** even if the database grows to 100,000 videos.
2.  **Automated Loop:**
    *   Load `benchmark_queries_10k.json` (generated in Part 0.5) — 10,000 queries, all with valid ground truth.
    *   Loop: `for (Query q : queries) { searchEngine.search(q.query); }`.
    *   Log per query: query text, augmentation type, latency (ms), Rank-1 VideoID, returned timestamp segments, and ground-truth hit (true/false at K=1,5,10).
3.  **Battery Isolation:**
    *   Keep USB connected for ADB throughout — `dumpsys batterystats` tracks per-app power draw regardless of charging state.
    *   For accurate mAh numbers, run on a device with a known starting charge level and note that USB charging partially offsets the draw. The per-UID CPU/NPU energy readings are still valid even while plugged in.

---

## Phase 4: Verification & Metrics Collection

### 1. Battery Consumption (mAh)
*   Reset stats: `adb shell dumpsys batterystats --reset`.
*   Run the full query benchmark (USB stays connected for ADB).
*   Dump report: `adb shell dumpsys batterystats > report.txt`.
*   Look for `Uid com.fedvcmr` to see total CPU/NPU power draw.
*   **Note:** USB charging partially offsets raw mAh draw, but per-UID CPU/NPU energy counters remain accurate and are the meaningful metric for comparing search engine efficiency.

### 2. Search Accuracy (Recall@K)
*   Pull the logs from the phone.
*   Use a Python script to compare the Rank-1 VideoID against ground truth from `benchmark_queries_10k.json`.
*   **Success Metric:** Recall@1 > 0.25 on ActivityNet is the baseline for a healthy MobileCLIP-S1 system.
*   Report Recall@1, Recall@5, Recall@10 **broken down by augmentation type**:
    *   `baseline` — measures raw retrieval quality
    *   `prefix_swap` — measures synonym robustness
    *   `ambiguity` — measures tolerance to vague/underspecified queries (hardest set; DGSE upgrade path applies here)
    *   `template_wrap` — measures whether surface framing (`"a video of..."`) duplicates or conflicts with the phone's internal prompt ensembling

### 3. Real-World Latency
*   Measure "Time-to-First-Frame" (TTFF) per query from the logs.
*   **Target: Sub-500ms** total latency including text encoding and full corpus search over ~8,000 frame embeddings.
*   Separate warm-cache latency (after LruCache hit) from cold-cache latency.

---

## Pre-Execution Checklist — Three Critical Reminders

Before running any phase of this plan, confirm the following three things. Everything else (filenames on the sdcard side, binary format, feature dimension, search engine logic) is correctly aligned with the working system.

**1. `phone_payload/index.json` is the wrong format and must not be pushed as `user_index.json`**
It uses the Charades-STA schema (`chunk_id`, `start`, `end`) which `SearchEngine.java` does not understand. The app will silently fail to load any videos — no crash, no error, just zero results. You must first generate a new `anet_user_index.json` from the ActivityNet extraction pipeline with the correct keys:
```
video_id, duration, num_frames, file_hash, ingested_at
```

**2. The adb push will permanently overwrite the working Charades-STA payload**
Run this backup command before any push:
```bash
adb pull /sdcard/fedvcmr/ ./phone_backup_charades/
```
Restore with `adb push ./phone_backup_charades/fedvcmr /sdcard/` if anything goes wrong.

**3. `num_frames: 8` must be written explicitly in every entry of `user_index.json`**
The Android app defaults to `num_frames = 16` when the field is absent. The Python extractor uses 8 frames. A missing field causes the app to read twice as many bytes per video from the blob — results will be completely wrong with no visible error. Every entry must include `"num_frames": 8`.

---

## Summary of Fixes vs. Original Plan

| Item | Original Plan | Actual State |
|---|---|---|
| Frames per video | 16 | Extractor uses **8**; Android defaults to 16 if `num_frames` missing from index — pack script must write `"num_frames": 8` explicitly |
| Blob size | ~32 MB | **~16 MB** |
| Index filename | `user_index.json` | **`features_index.json`** (from pack script) |
| val_1.json query count | "10,000 queries" | 733-video corpus → 2,638 ground-truth base queries → **10k via NLP augmentation** (prefix swap, ambiguity injection, template wrap) |
| Download split | implied val | **download_anet.py targets train split** — must fix |
| Prompt ensembling | "must implement" | **Already implemented** (6 templates) |
| Hubness suppression | not mentioned | **Already implemented** |
| Temporal smoothing | not mentioned | **Already implemented** |
| LruCache text cache | "must implement" | **Not yet implemented** — still a TODO |
| 2D dot product matrix | "must implement" | **Partially** — inline only, no precomputed matrix |
| BenchmarkService.java | implied complete | **Stub only** — loop body missing |
| Battery test (USB) | "run unplugged" | USB stays connected; per-UID CPU/NPU energy counters are accurate even while charging |
