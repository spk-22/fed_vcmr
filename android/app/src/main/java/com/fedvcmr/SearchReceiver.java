package com.fedvcmr;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.util.Base64;
import android.util.Log;

import com.fedvcmr.core.SearchEngine;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * VCMR 3-stage search pipeline (thin wrapper around SearchEngine).
 *
 * Trigger from PC:
 *   adb shell am broadcast -a com.fedvcmr.SEARCH --es query "cooking pasta"
 *
 * Remote Ingest:
 *   adb shell am broadcast -a com.fedvcmr.INGEST --es path "/sdcard/..."
 */
public class SearchReceiver extends BroadcastReceiver {

    private static final String TAG = "VCMR_RESULT";
    private static SearchEngine cachedEngine;
    private static final ExecutorService ingestExecutor = Executors.newSingleThreadExecutor();

    @Override
    public void onReceive(Context context, Intent intent) {
        final String action = intent.getAction();
        final Context ctx = context.getApplicationContext();
        
        if ("com.fedvcmr.SEARCH".equals(action)) {
            final String queryB64 = intent.getStringExtra("query_emb");
            final String queryText = intent.getStringExtra("query");
            new Thread(() -> runSearch(ctx, queryB64, queryText)).start();
        } else if ("com.fedvcmr.INGEST".equals(action)) {
            final String path = intent.getStringExtra("path");
            if (path != null) {
                ingestExecutor.execute(() -> runIngest(ctx, path));
            }
        }
    }

    private void runIngest(Context ctx, String path) {
        android.os.PowerManager pm = (android.os.PowerManager) ctx.getSystemService(Context.POWER_SERVICE);
        android.os.PowerManager.WakeLock wakeLock = pm.newWakeLock(android.os.PowerManager.PARTIAL_WAKE_LOCK, "fedvcmr:IngestLock");
        
        try {
            wakeLock.acquire(10 * 60 * 1000L); // 10 min max
            if (cachedEngine == null) {
                cachedEngine = new SearchEngine(ctx);
            }
            Log.i(TAG, "Remote ingest requested: " + path);
            VideoIngestor.ingest(ctx, Uri.fromFile(new java.io.File(path)), cachedEngine, new VideoIngestor.Callback() {
                @Override public void onProgress(String message) { Log.d(TAG, "Ingest: " + message); }
                @Override public void onSuccess(String videoId, int chunks, int total) {
                    Log.i(TAG, "Ingest SUCCESS: " + videoId + " (Total videos: " + total + ")");
                }
                @Override public void onError(String error) { Log.e(TAG, "Ingest ERROR: " + error); }
            });
        } catch (Exception e) {
            Log.e(TAG, "Remote ingest failed", e);
        } finally {
            if (wakeLock.isHeld()) wakeLock.release();
        }
    }

    private void runSearch(Context ctx, String queryB64, String queryText) {
        try {
            if (cachedEngine == null) {
                cachedEngine = new SearchEngine(ctx);
            }

            if (queryText != null && !queryText.isEmpty()) {
                Log.i(TAG, "Searching for: " + queryText);
                List<SearchEngine.Hit> results = cachedEngine.search(queryText);
                logResults(queryText, results);

            } else if (queryB64 != null && !queryB64.isEmpty()) {
                byte[] queryBytes = Base64.decode(queryB64, Base64.DEFAULT);
                float[] queryEmb = new float[512];
                ByteBuffer.wrap(queryBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(queryEmb);
                Log.i(TAG, "Received base64-encoded query embedding (legacy)");
            } else {
                Log.e(TAG, "No query provided.");
            }

        } catch (Exception e) {
            Log.e(TAG, "Search failed: " + e.getMessage(), e);
        }
    }

    private void logResults(String query, List<SearchEngine.Hit> results) {
        if (results.isEmpty()) {
            Log.i(TAG, String.format("query=\"%s\" | results=0 | no matches", query));
            return;
        }

        for (int i = 0; i < Math.min(3, results.size()); i++) {
            SearchEngine.Hit hit = results.get(i);
            Log.i(TAG, String.format(
                "query=\"%s\" | rank=%d | video=%s | segment=%.1fs-%.1fs | score=%.4f",
                query, i + 1, hit.videoId, hit.tStart, hit.tEnd, hit.score
            ));
        }

        try {
            SearchEngine.Hit winner = results.get(0);
            saveEvidenceFrame(winner, query);
        } catch (Exception e) {
            Log.e(TAG, "Failed to save evidence frame: " + e.getMessage());
        }
    }

    private void saveEvidenceFrame(SearchEngine.Hit hit, String query) throws Exception {
        String verificationDir = "/sdcard/fedvcmr/verification";
        new java.io.File(verificationDir).mkdirs();

        android.media.MediaMetadataRetriever mmr = new android.media.MediaMetadataRetriever();
        String videoPath = "/sdcard/fedvcmr/user_videos/" + hit.videoId + ".mp4";
        
        try {
            mmr.setDataSource(videoPath);
            long timeUs = (long)(hit.tStart * 1_000_000L);
            android.graphics.Bitmap frame = mmr.getFrameAtTime(timeUs, android.media.MediaMetadataRetriever.OPTION_CLOSEST);
            
            if (frame != null) {
                String safeQuery = query.replaceAll("[^a-zA-Z0-9]", "_").toLowerCase();
                if (safeQuery.length() > 30) safeQuery = safeQuery.substring(0, 30);
                String fileName = String.format("evidence_%s_%s.jpg", safeQuery, hit.videoId);
                java.io.File outFile = new java.io.File(verificationDir, fileName);
                
                try (java.io.FileOutputStream out = new java.io.FileOutputStream(outFile)) {
                    frame.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out);
                    Log.i(TAG, "SAVED EVIDENCE: " + outFile.getAbsolutePath());
                }
                frame.recycle();
            }
        } finally {
            mmr.release();
        }
    }
}
