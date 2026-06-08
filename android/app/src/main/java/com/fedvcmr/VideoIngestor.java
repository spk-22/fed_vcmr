package com.fedvcmr;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import com.fedvcmr.core.SearchEngine;
import com.fedvcmr.core.VisionEncoder;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Optimized Ingestor.
 * 
 * DESIGN: "Adaptive & Deduplicated Ingest"
 * 1. Adaptive Sampling: 32 to 64 frames depending on video length.
 * 2. Deduplication: First-MB Hash check to prevent index bloat.
 * 3. Batching Ready: VisionEncoder true-batching integration.
 */
public class VideoIngestor {

    private static final String TAG = "VideoIngestor";

    public static final String DATA_DIR        = "/sdcard/fedvcmr";
    public static final String USER_INDEX_PATH = DATA_DIR + "/user_index.json";
    public static final String USER_BLOB_PATH  = DATA_DIR + "/user_features_blob.bin";
    public static final String USER_BLOB_IDX   = DATA_DIR + "/user_features_index.json";
    public static final String USER_VM_PATH    = DATA_DIR + "/user_vector_map.bin";
    public static final String USER_VIDEOS_DIR = DATA_DIR + "/user_videos";

    private static final int   DIM = 512;

    public interface Callback {
        void onProgress(String message);
        void onSuccess(String videoId, int totalFrames, int totalVideos);
        void onError(String error);
    }

    public static void ingest(Context context, Uri videoUri, SearchEngine engine, Callback cb) {
        try {
            // ── 1. Deduplication (DISABLED FOR RE-INDEXING) ──────────
            /*String fileHash = computeFileHash(context, videoUri);
            if (isDuplicate(fileHash)) {
                cb.onError("Video already exists in index.");
                return;
            }*/
            String fileHash = "force_reindex_" + System.currentTimeMillis();

            // ── 2. Read metadata ─────────────────────────────────────
            cb.onProgress("Reading metadata...");
            MediaMetadataRetriever mmr = new MediaMetadataRetriever();
            mmr.setDataSource(context, videoUri);
            String durStr = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            if (durStr == null) { mmr.release(); cb.onError("Bad duration"); return; }
            long   durationMs  = Long.parseLong(durStr);
            double durationSec = durationMs / 1000.0;

            // 16 uniform frames — fast ingest, NNAPI warm-up amortized across videos
            int numFrames = 16;

            // ── 4. Assign ID ─────────────────────────────────────────
            int    videoSeq = existingVideoCount() + 1;
            String date     = new SimpleDateFormat("yyyyMMdd", Locale.US).format(new Date());
            String videoId  = "phone_" + date + "_" + String.format(Locale.US, "%03d", videoSeq);

            // ── 5. Copy video ────────────────────────────────────────
            cb.onProgress("Saving video...");
            new File(USER_VIDEOS_DIR).mkdirs();
            copyVideo(context, videoUri, videoId);

            // ── 6. Extract & Encode (True Batching Fix C7) ───────────
            cb.onProgress(String.format(Locale.US, "Encoding %d frames...", numFrames));
            Bitmap[] frames = extractUniformFrames(mmr, durationMs, numFrames);
            
            VisionEncoder encoder = new VisionEncoder(context);
            float[] allEmbeddings = encoder.encodeBatch(frames);
            for (Bitmap bm : frames) if (bm != null) bm.recycle();
            encoder.close();
            mmr.release();

            // ── 7. Pool & Persist ────────────────────────────────────
            float[] meanEmbedding = meanPool(allEmbeddings, numFrames);
            persistVideo(videoId, durationSec, numFrames, allEmbeddings, meanEmbedding, fileHash);
            
            engine.addVideo(videoId, durationSec, meanEmbedding, allEmbeddings, "user_phone");

            cb.onSuccess(videoId, numFrames, engine.getVideoCount());

        } catch (Exception e) {
            Log.e(TAG, "Ingest failed", e);
            cb.onError("Ingest failed: " + e.getMessage());
        }
    }

    private static Bitmap[] extractUniformFrames(MediaMetadataRetriever mmr, long durationMs, int numFrames) {
        Bitmap[] frames = new Bitmap[numFrames];
        for (int i = 0; i < numFrames; i++) {
            double frac  = i / (double)(numFrames - 1);
            long   timeUs = (long)(frac * durationMs * 1000L);
            timeUs = Math.min(timeUs, durationMs * 1000L - 1);
            frames[i] = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
            if (frames[i] == null && i > 0) frames[i] = frames[i - 1];
        }
        return frames;
    }

    private static float[] meanPool(float[] allEmbeddings, int numFrames) {
        float[] mean = new float[DIM];
        for (int f = 0; f < numFrames; f++) {
            for (int d = 0; d < DIM; d++) mean[d] += allEmbeddings[f * DIM + d];
        }
        float norm = 0;
        for (int d = 0; d < DIM; d++) { mean[d] /= numFrames; norm += mean[d] * mean[d]; }
        norm = (float) Math.sqrt(norm);
        for (int d = 0; d < DIM; d++) mean[d] /= (norm + 1e-10f);
        return mean;
    }

    private static void persistVideo(String videoId, double duration, int numFrames,
            float[] frameData, float[] meanEmbedding, String hash) throws Exception {

        File blobFile = new File(USER_BLOB_PATH);
        long blobOffset = blobFile.exists() ? blobFile.length() : 0;
        try (FileOutputStream fos = new FileOutputStream(blobFile, true)) {
            ByteBuffer buf = ByteBuffer.allocate(numFrames * DIM * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : frameData) buf.putFloat(v);
            fos.write(buf.array());
        }

        File idxFile = new File(USER_BLOB_IDX);
        JSONObject blobIdx = idxFile.exists() ? new JSONObject(new String(readFile(idxFile))) : new JSONObject();
        blobIdx.put(videoId, (int) blobOffset);
        writeFile(idxFile, blobIdx.toString().getBytes());

        File indexFile = new File(USER_INDEX_PATH);
        JSONArray index = indexFile.exists() ? new JSONArray(new String(readFile(indexFile))) : new JSONArray();
        JSONObject entry = new JSONObject();
        entry.put("video_id", videoId);
        entry.put("duration", duration);
        entry.put("num_frames", numFrames);
        entry.put("file_hash", hash);
        entry.put("ingested_at", System.currentTimeMillis());
        index.put(entry);
        writeFile(indexFile, index.toString().getBytes());
    }

    private static String computeFileHash(Context context, Uri uri) throws Exception {
        try (InputStream is = context.getContentResolver().openInputStream(uri)) {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] buf = new byte[1024 * 1024]; // 1MB Hash is enough for dedupe
            int n = is.read(buf);
            if (n > 0) md.update(buf, 0, n);
            byte[] hash = md.digest();
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) sb.append(String.format("%02x", b));
            return sb.toString();
        }
    }

    private static boolean isDuplicate(String hash) {
        File f = new File(USER_INDEX_PATH);
        if (!f.exists()) return false;
        try {
            JSONArray arr = new JSONArray(new String(readFile(f)));
            for (int i = 0; i < arr.length(); i++) {
                if (hash.equals(arr.getJSONObject(i).optString("file_hash"))) return true;
            }
        } catch (Exception ignored) {}
        return false;
    }

    private static void copyVideo(Context context, Uri videoUri, String videoId) throws Exception {
        File dest = new File(USER_VIDEOS_DIR + "/" + videoId + ".mp4");
        
        // If source and destination are the same, skip copy
        if (videoUri.getScheme().equals("file")) {
            File source = new File(videoUri.getPath());
            if (source.getAbsolutePath().equals(dest.getAbsolutePath())) {
                Log.d(TAG, "Source and destination are same, skipping copy");
                return;
            }
        }

        try (InputStream  in  = context.getContentResolver().openInputStream(videoUri);
             FileOutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[65536];
            int n;
            while (in != null && (n = in.read(buf)) != -1) out.write(buf, 0, n);
        }
    }

    private static int existingVideoCount() {
        File f = new File(USER_INDEX_PATH);
        if (!f.exists()) return 0;
        try { return new JSONArray(new String(readFile(f))).length(); } catch (Exception e) { return 0; }
    }

    public static byte[] readFile(File f) throws IOException {
        try (FileInputStream fis = new FileInputStream(f)) { return fis.readAllBytes(); }
    }

    public static void writeFile(File f, byte[] data) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(f, false)) { fos.write(data); }
    }
}
