package com.fedvcmr.core;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;
import android.util.LruCache;

import com.fedvcmr.VideoIngestor;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * VCMR Engine with "Coded" DGSE and Temporal Grounding.
 * Uses Hubness Suppression and Temporal Smoothing instead of TFLite models.
 */
public class SearchEngine {
    private static final String TAG = "SearchEngine";

    private static final int DIM    = 512;
    private static final int TOP_K_VIDEOS = 10;
    private static final float HUB_LAMBDA = 0.35f; // Penalize "popular" videos

    private Context context;
    private TextEncoder textEncoder;
    private final LruCache<String, float[]> queryCache = new LruCache<>(1000);
    
    public static class VideoData {
        public String videoId;
        public double duration;
        public float[] frameEmbeddings; 
        public int numFrames;
        public String source;
        public float meanSim; // Global "hubness" of this video
    }

    private final List<VideoData> videos = new ArrayList<>();

    public static class Hit {
        public String videoId;
        public double tStart, tEnd;
        public float  score;
        public String chunkId;
        public String sourceDataset;
        public boolean fused = false;

        public Hit(String videoId, double tStart, double tEnd,
                   float score, String chunkId, String sourceDataset) {
            this.videoId       = videoId;
            this.tStart        = tStart;
            this.tEnd          = tEnd;
            this.score         = score;
            this.chunkId       = chunkId;
            this.sourceDataset = sourceDataset;
        }
    }

    public SearchEngine(Context context) throws Exception {
        this.context = context.getApplicationContext();
        loadUserData();
    }

    public synchronized List<Hit> search(String queryStr) throws Exception {
        long t0 = SystemClock.elapsedRealtimeNanos();
        ensureModelsLoaded();
        if (videos.isEmpty()) return Collections.emptyList();

        // --- C4: Query Cache ---
        float[] qFinal = queryCache.get(queryStr);
        if (qFinal == null) {
            long tEnc0 = SystemClock.elapsedRealtime();
            // --- A1: Reduced Ensembling for Latency ---
            String[] templates = {"%s", "a video of %s"};
            float[] qEnsemble = new float[DIM];
            for (String temp : templates) {
                float[] emb = textEncoder.encode(String.format(temp, queryStr));
                for (int i = 0; i < DIM; i++) qEnsemble[i] += emb[i];
            }
            qFinal = normalize(qEnsemble);
            queryCache.put(queryStr, qFinal);
            Log.v(TAG, "Encoding time: " + (SystemClock.elapsedRealtime() - tEnc0) + "ms");
        }

        long tSearch0 = SystemClock.elapsedRealtime();
        // --- C2: Precompute all dot products ---
        int numV = videos.size();
        float[][] allScores = new float[numV][];
        float globalSum = 0;
        int totalFrames = 0;
        
        for (int i = 0; i < numV; i++) {
            VideoData v = videos.get(i);
            allScores[i] = new float[v.numFrames];
            float vSum = 0;
            for (int f = 0; f < v.numFrames; f++) {
                allScores[i][f] = dot(qFinal, v.frameEmbeddings, f * DIM);
                vSum += allScores[i][f];
            }
            v.meanSim = vSum / v.numFrames;
            globalSum += vSum;
            totalFrames += v.numFrames;
        }
        float globalMean = globalSum / (float)Math.max(1, totalFrames);

        // 3. Score and Ground
        List<Hit> hits = new ArrayList<>();
        for (int i = 0; i < numV; i++) {
            VideoData v = videos.get(i);
            float[] smoothedScores = new float[v.numFrames];
            float maxVal = -1e9f;
            int maxIdx = 0;

            for (int f = 0; f < v.numFrames; f++) {
                float sum = allScores[i][f];
                int count = 1;
                if (f > 0) { sum += allScores[i][f-1]; count++; }
                if (f < v.numFrames - 1) { sum += allScores[i][f+1]; count++; }
                smoothedScores[f] = sum / count;
                float finalScore = smoothedScores[f] - (HUB_LAMBDA * (v.meanSim - globalMean));
                if (finalScore > maxVal) { maxVal = finalScore; maxIdx = f; }
            }
            double dPF = v.duration / (double)(Math.max(1, v.numFrames - 1));
            hits.add(groundMomentCoded(v, smoothedScores, maxIdx, maxVal, dPF));
        }

        Collections.sort(hits, (a, b) -> Float.compare(b.score, a.score));
        List<Hit> results = hits.subList(0, Math.min(TOP_K_VIDEOS, hits.size()));

        long totalLatency = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000;
        if (!results.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (int k = 0; k < Math.min(3, results.size()); k++) {
                sb.append(String.format("[%s:%.3f] ", results.get(k).videoId, results.get(k).score));
            }
            Log.i(TAG, String.format("VCMR: query=\"%s\" | top3=%s | lat=%dms",
                    queryStr, sb.toString(), totalLatency));
        }
        return results;
    }

    /** 
     * Refined Grounding using the smoothed score curve.
     */
    private Hit groundMomentCoded(VideoData v, float[] scores, int maxIdx, float maxScore, double dPF) {
        float threshold = scores[maxIdx] * 0.92f; // Tighter threshold for smoothed curve
        int startIdx = maxIdx, endIdx = maxIdx;

        while (startIdx > 0 && scores[startIdx-1] >= threshold) startIdx--;
        while (endIdx < v.numFrames-1 && scores[endIdx+1] >= threshold) endIdx++;

        // Minimum 2.5s segment
        while ((endIdx - startIdx) * dPF < 2.5 && (startIdx > 0 || endIdx < v.numFrames-1)) {
            if (startIdx > 0) startIdx--;
            if (endIdx < v.numFrames-1) endIdx++;
        }

        return new Hit(v.videoId, startIdx * dPF, endIdx * dPF, maxScore, v.videoId, v.source);
    }

    private void loadUserData() {
        try {
            File idxFile = new File(VideoIngestor.USER_INDEX_PATH);
            File bFile   = new File(VideoIngestor.USER_BLOB_PATH);
            File biFile  = new File(VideoIngestor.USER_BLOB_IDX);
            if (!idxFile.exists() || !bFile.exists() || !biFile.exists()) return;
            
            JSONArray index = new JSONArray(new String(VideoIngestor.readFile(idxFile)));
            JSONObject offsets = new JSONObject(new String(VideoIngestor.readFile(biFile)));
            
            // --- MMAP: Open the feature blob as a memory-mapped file ---
            java.io.RandomAccessFile raf = new java.io.RandomAccessFile(bFile, "r");
            java.nio.channels.FileChannel channel = raf.getChannel();
            java.nio.MappedByteBuffer mmap = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size());
            mmap.order(ByteOrder.LITTLE_ENDIAN);

            for (int i = 0; i < index.length(); i++) {
                JSONObject entry = index.getJSONObject(i);
                VideoData v = new VideoData();
                v.videoId = entry.getString("video_id");
                v.duration = entry.getDouble("duration");
                v.source = entry.optString("source_dataset", "user_phone");
                int off = offsets.getInt(v.videoId);
                v.numFrames = entry.optInt("num_frames", 16);
                
                v.frameEmbeddings = new float[v.numFrames * DIM];
                // Read from MMAP buffer instead of heap byte[]
                mmap.position(off);
                mmap.asFloatBuffer().get(v.frameEmbeddings);
                
                // --- Force L2 Normalization (Rule 1 Fix) ---
                for (int f = 0; f < v.numFrames; f++) {
                    float norm = 0;
                    for (int d = 0; d < DIM; d++) {
                        float val = v.frameEmbeddings[f * DIM + d];
                        norm += val * val;
                    }
                    norm = (float) Math.sqrt(norm);
                    if (norm > 1e-6f) {
                        for (int d = 0; d < DIM; d++) v.frameEmbeddings[f * DIM + d] /= norm;
                    }
                }
                
                videos.add(v);
            }
            raf.close();
            Log.i(TAG, "MMAP Loaded " + videos.size() + " videos into Optimized Engine");
        } catch (Exception e) { Log.e(TAG, "Load failed", e); }
    }

    private void ensureModelsLoaded() throws Exception {
        if (textEncoder == null) textEncoder = new TextEncoder(context);
    }

    public void close() {
        if (textEncoder != null) textEncoder.close();
    }

    private float dot(float[] a, float[] b, int bOff) {
        float sum = 0; 
        for (int i = 0; i < DIM; i++) sum += a[i] * b[bOff + i];
        return sum;
    }

    private float[] normalize(float[] v) {
        float norm = 0; 
        for (float x : v) norm += x * x;
        norm = (float) Math.sqrt(norm);
        if (norm < 1e-10f) return v;
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / norm;
        return out;
    }

    public void addVideo(String videoId, double duration, float[] mean, float[] frames, String src) {
        VideoData v = new VideoData();
        v.videoId = videoId; v.duration = duration; v.frameEmbeddings = frames;
        v.numFrames = frames.length / DIM; v.source = src;
        videos.add(v);
    }
    public int getVideoCount() { return videos.size(); }
}
