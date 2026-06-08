# On-Device VCMR Rewamp Progress

## Phase 0: Establish Ground Truth on Device
- [x] Export PC embeddings for test videos (Charades/ActivityNet)
- [x] Dump phone embeddings for matched timestamps
- [x] Compute cosine similarity gap analysis (Identified 0.55-0.92 sim due to frame extraction drift)
- [x] Validate text encoder/tokenizer alignment (Identified missing Text Projection Head in Android)

## Phase 1: Fix Frame Extraction (Highest ROI)
- [x] Locate `VideoIngestor.extractFrames()` in Android source
- [x] Replace `OPTION_CLOSEST_SYNC` with `OPTION_CLOSEST` to ensure unique frames are extracted
- [x] Verify unique frames are being extracted instead of keyframe duplicates

## Phase 2: TSN-style Video-level Sampling
- [x] Implement 32-frame uniform sampling across video duration
- [x] Remove sliding window index primitives
- [x] Update index structure to store video-level blocks (TSN approach)

## Phase 3: Simplify Retrieval Pipeline
- [x] Remove DGSE reranker from on-device search path
- [x] Remove learned temporal grounding model from search path
- [x] Implement Multi-Moment Peak-Detection grounding
- [x] Implement greedy grounding expansion (80% peak threshold + 3s minimum duration)

## Phase 4: Fix Text-Vision Alignment
- [x] Verify MobileCLIP checkpoint consistency across vision/text/heads
- [x] Validate image preprocessing normalization (Fixed ImageNet-vs-CLIP drift in VisionEncoder)
- [x] Implement NLP Expansion: Prompt Ensembling (Mean of 5 templates)
- [x] Port Text Projection Head to Android TextEncoder (Integrated TFLite head interpreter)

## Phase 5: Personal Calibration Layer
- [ ] Implement user-interaction logging (positive/negative pairs)
- [ ] Implement orthogonal alignment matrix (Procrustes) calculation
- [x] Implement Standardized Relative Scoring (Z-Norm) to solve Video Pollution/Bias.

## Phase 6: Incremental Ingest & Refactoring
- [x] Transition to video-level index structure in SearchEngine
- [x] Updated VideoIngestor to support video-level persistence and sampling
- [x] Fixed UI (MainActivity) to support video-level API and search latency tracking

---

# Current System Status (As of April 19, 2026)

### ✅ What is Working
1.  **Stable Retrieval (Z-Norm):** The search engine now uses Z-Score Normalization. Instead of absolute cosine scores, it ranks moments by how many standard deviations they stand out from your specific library's "noise."
2.  **Beach/Pushup Ranking Fix:** High-contrast videos (like pushups or jumping) no longer "pollute" every search. The system correctly identifies that "Beach" is the unique match for a beach query, even if "Pushups" has higher raw CLIP contrast.
3.  **Corrected Sampler:** Video ingestion is now 4x faster and extracts unique frames rather than repeating keyframes.
4.  **Mathematical Alignment:** Text and Vision are finally "speaking the same language" thanks to the TFLite Text Projection Head.
5.  **Robust UI:** 
    *   Manual ingestion via the `+` button works from any folder.
    *   Search latency is clearly displayed (~2.5s for a 5-video index).
    *   Multi-moment results are correctly displayed in the list.

### ⚠️ What is NOT Working / Known Issues
1.  **Playback Stability:** Video playback in the results list occasionally fails or stays silent if the Android `ExoPlayer` cannot immediately acquire the file handle for newly ingested videos.
2.  **Index Redundancy:** `VideoIngestor` currently only appends to the index. If you ingest the same video twice, it will appear twice in the search results.
3.  **Standard Deviation Sensitivity:** On extremely small indexes (fewer than 3 videos), Z-Norm can be statistically unstable. The system becomes significantly more accurate once 5+ videos are indexed.
4.  **Temporal Resolution:** Because we use 32 frames for the whole video, the "start/end" timestamps for very long videos (>2 mins) are accurate only to within ~4-5 seconds.

### 📊 User Video Ranking Performance
*   **Query: "Exercises on the floor"** → **Rank 1: `phone_004` (Correct)**.
*   **Query: "Walking on the beach"** → **Rank 1: `phone_003` (Correct)**.
*   **Query: "Driving"** → **Rank 1: `phone_005` (Correct)**.
*   *Observation:* Previously, "Pushups" would win every query. Now, the Z-Norm successfully isolates the specific action.
