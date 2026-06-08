# FedVCMR Android — Session Context

## Project Overview
On-device Video Corpus Moment Retrieval (VCMR) system running on Android using MobileCLIP-S1. The app ingests user videos, encodes them with a vision encoder, and retrieves video segments matching a text query — all on-device.

**Device:** OnePlus (Snapdragon 8 Gen 2, Hexagon 780 NPU)  
**Working directory:** `C:\prism\android`  
**Package:** `com.fedvcmr`

---

## Architecture

### Search Pipeline
1. User sends text query via adb broadcast: `adb shell am broadcast -a com.fedvcmr.SEARCH --es query "..."`
2. `SearchReceiver.java` receives it, spins a background thread
3. `SearchEngine.java` encodes the query via `TextEncoder`, scores all indexed videos, returns top-K hits
4. Results logged as: `query="..." | rank=N | video=... | segment=Xs-Ys | score=F`

### Text Encoding (TextEncoder.java)
Two-stage pipeline:
- **Stage 1:** PyTorch Mobile backbone (`mobileclip_text_encoder.ptl`) — takes token IDs `[1, 77]`, outputs raw 512-dim features
- **Stage 2:** TFLite projection head (`text_head_best_model.tflite`) — projects 512→512
- **Output:** L2-normalized 512-dim embedding

Tokenizer: `CLIPTokenizer` in Java using `clip_encoder.json` + `clip_bpe_ranks.json` + `clip_byte_encoder.json`  
Verified tokens: person=2533, sitting=4919, stairs=9577 (matches Python open_clip MobileCLIP-S1)

### Vision Encoding (VisionEncoder.java)
Two-stage pipeline:
- **Stage 1:** TFLite backbone (`mobileclip_vision_encoder.tflite`) via NNAPI delegate (Hexagon 780 NPU), fixed input shape `[1, 256, 256, 3]` float32 NHWC
- **Stage 2:** TFLite projection head (`vision_head_best_model.tflite`) on CPU
- **Output:** L2-normalized 512-dim embedding per frame
- Pixel normalization: `/255` only (mean=0, std=1 — correct for MobileCLIP-S1)

### Ingest Pipeline (VideoIngestor.java)
- **16 uniform frames** per video using `OPTION_CLOSEST_SYNC` (fast keyframe extraction ~3s/video after NNAPI warmup)
- Frame embeddings stored in `/sdcard/fedvcmr/user_features_blob.bin` (flat float32, little-endian)
- Index at `/sdcard/fedvcmr/user_index.json`, offsets at `/sdcard/fedvcmr/user_features_index.json`
- Deduplication disabled (force_reindex mode)

### SearchEngine.java
- **Hub correction:** `corrected = s - 0.3 * (vMeans[i] - globalMean)` (coefficient reduced from 0.7 to 0.3)
- **Ranking:** ArgMax — rank videos by single best frame-to-query cosine similarity
- **Grounding (`groundMoment`):** Greedy expansion from argmax frame using threshold `= maxScore * 0.85`, minimum 3s segment enforced
- **Top-K:** Returns top 10 videos

---

## Model Assets (in `android/app/src/main/assets/models/`)

| File | Size | Purpose |
|---|---|---|
| `mobileclip_text_encoder.ptl` | 178.5 MB | Text backbone (newly re-exported, INT8 quantized) |
| `text_head_best_model.tflite` | ~1 MB | FL-trained text projection head |
| `mobileclip_vision_encoder.tflite` | ~80 MB | Vision backbone (NNAPI fixed shape [1,256,256,3]) |
| `vision_head_best_model.tflite` | ~1 MB | FL-trained vision projection head |
| `dgse.tflite` | small | DGSE model |
| `temporal_grounding.tflite` | small | Temporal grounding transformer |

Other assets (root of assets/): `clip_encoder.json`, `clip_bpe_ranks.json`, `clip_byte_encoder.json`

---

## Key History & Bugs Fixed This Session

### Bug 1: Broken Text Backbone PTL (ROOT CAUSE of all low scores)
**Symptom:** All search scores 0.009–0.077 regardless of query content. Queries 3 and 4 returning same video.  
**Diagnosis:** Loaded `mobileclip_text_encoder.ptl` in Python and compared output to open_clip reference. Cosine similarity = **0.49** (should be 1.0). The PTL was a broken export — accepted correct input shape but produced garbage output vectors.  
**Fix:** Re-exported from `open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')` using `model.text` (the `TextTransformer`), traced with `torch.jit.trace(..., strict=False, check_trace=False)`, applied dynamic INT8 quantization on Linear layers (254MB → 178.5MB), saved with `_save_for_lite_interpreter`.  
**Result after fix:** Query "woman sitting on stairs with a laptop" score jumped from 0.009 → **0.1984**.

### Bug 2: Wrong Vocabulary (clip_encoder.json) — Fixed in prior session
**Symptom:** Token IDs for common words were wrong (person=1859 instead of 2533).  
**Fix:** Dumped vocabulary from `open_clip.get_tokenizer('MobileCLIP-S1')` and replaced `clip_encoder.json`.  
**Verified:** person=2533, sitting=4919, stairs=9577 now match Python reference.

### Bug 3: groundMoment() Deleted — Fixed in prior session
**Symptom:** Segment start and end were identical (tStart=tEnd=maxIdx*dPF), no actual temporal grounding.  
**Fix:** Restored full expansion logic using query-vector threshold comparison.

### Bug 4: Ingest too slow (60s/video) — Fixed in prior session
**Fix:** Changed from adaptive 32–64 frames with `OPTION_CLOSEST` to flat 16 frames with `OPTION_CLOSEST_SYNC`.

### Bug 5: Hub correction too aggressive — Fixed in prior session
**Fix:** Reduced λ from 0.7 to 0.3 in hub penalty formula.

---

## PC Verification Workflow

**Script:** `C:/prism/scripts/verify_query.sh "query text"`  
- Fires adb broadcast, waits for logcat, parses results, pulls video, extracts midpoint JPG

**Evidence frames:** Saved by app itself to `/sdcard/fedvcmr/verification/evidence_*.jpg`, pulled via adb to `C:/prism/pc_verification/verification/`

**Python diagnostic:** `C:/prism/scripts/_extract_verification.py` — parses logcat rank/video/segment, pulls video via adb, extracts frame with OpenCV

---

## Current Test Results (after text PTL fix, 12 ingested Charades videos)

| Query | Score | Top-1 Video | Segment | Visual Result |
|---|---|---|---|---|
| woman sitting on stairs with a laptop | 0.1984 | phone_20260419_013 | 22.8s–31.1s | Man in kitchen — WRONG |
| two people sitting on a sofa | 0.0412 | phone_20260419_011 | 24.4s–28.1s | Man alone in office chair — WRONG |
| person washing dishes at kitchen sink | 0.0221 | phone_20260419_009 | 0.0s–3.6s | Empty bathroom — WRONG |
| person in a bathroom | -0.0041 | phone_20260419_010 | 22.0s–33.0s | Person at kitchen sink — WRONG (ironic swap) |

**Scores 2–4 are still near-zero.** Query 1 improved dramatically from the PTL fix. Queries 3 and 4 are effectively swapped (bathroom video ranked for "washing dishes", kitchen sink ranked for "bathroom").

---

## Diagnosis: Remaining Issue

The `text_head_best_model.tflite` and `vision_head_best_model.tflite` are FL-trained (Federated Learning) projection heads. These heads appear to **not be jointly trained with a cross-modal contrastive objective** — meaning text head output and vision head output are in different embedding spaces. MobileCLIP's backbone itself already aligns text and vision in a shared space (that's the core CLIP training objective).

**PC diagnostic run** (`scripts/diag_alignment.py`) against 400-video Charades phone_payload confirmed:
- RAW backbone mode (no heads): expected-video scores 0.12–0.17, top-1 scores 0.26–0.31
- BOTH_HEADS mode: expected-video scores 0.07–0.27, top-1 scores 0.43–0.49 (extreme hub effect)

The heads inflate certain video scores to 0.43–0.49 regardless of query, classic hub behaviour from non-joint training.

---

## File Locations

| Path | Description |
|---|---|
| `C:/prism/android/` | Android project root |
| `C:/prism/android/app/src/main/java/com/fedvcmr/core/SearchEngine.java` | Main retrieval logic |
| `C:/prism/android/app/src/main/java/com/fedvcmr/core/TextEncoder.java` | Text encoding pipeline |
| `C:/prism/android/app/src/main/java/com/fedvcmr/core/VisionEncoder.java` | Vision encoding pipeline |
| `C:/prism/android/app/src/main/java/com/fedvcmr/VideoIngestor.java` | Video ingest pipeline |
| `C:/prism/android/app/src/main/java/com/fedvcmr/SearchReceiver.java` | ADB broadcast receiver |
| `C:/prism/checkpoints/export/` | All exported model files |
| `C:/prism/checkpoints/export/mobileclip_text_encoder_v2.ptl` | Correct float32 text backbone (254MB) |
| `C:/prism/checkpoints/export/mobileclip_text_encoder_q8.ptl` | INT8 quantized text backbone (178.5MB, deployed) |
| `C:/prism/scripts/export_text_ptl.py` | Script to re-export text backbone PTL |
| `C:/prism/scripts/quantize_text_ptl.py` | Script to quantize PTL to INT8 |
| `C:/prism/scripts/diag_alignment.py` | PC diagnostic: text-vision alignment check |
| `C:/prism/scripts/verify_query.sh` | Shell script for visual verification workflow |
| `C:/prism/scripts/_extract_verification.py` | Parse logcat + pull frames via adb |
| `C:/prism/pc_verification/verification/` | Output JPG frames for visual inspection |
| `/sdcard/fedvcmr/user_index.json` | Phone: video index |
| `/sdcard/fedvcmr/user_features_blob.bin` | Phone: frame embeddings blob |
| `/sdcard/fedvcmr/user_features_index.json` | Phone: blob byte offsets |
| `/sdcard/fedvcmr/user_videos/` | Phone: ingested video files |
| `/sdcard/fedvcmr/verification/` | Phone: evidence JPGs saved by app |

---

## Python Environment
- Virtual env: `C:/prism/.venv`
- Run scripts as: `.venv/Scripts/python scripts/...`
- Key packages: `open_clip`, `torch`, `tensorflow`, `opencv-python`, `onnx`, `onnx2tf`

---

## Build & Deploy
```bash
cd C:/prism/android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Wipe Index (before re-ingest)
```bash
adb shell rm -f //sdcard/fedvcmr/user_index.json //sdcard/fedvcmr/user_features_blob.bin //sdcard/fedvcmr/user_features_index.json //sdcard/fedvcmr/user_vector_map.bin
```

## Launch App
```bash
adb shell am start -n com.fedvcmr/.ui.MainActivity
```

## Run Query
```bash
adb shell am broadcast -a com.fedvcmr.SEARCH --es query "your query here"
# Wait ~30s for first query (model load), then check:
adb logcat -d | grep VCMR_RESULT
```
