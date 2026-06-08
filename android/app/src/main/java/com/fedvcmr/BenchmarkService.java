package com.fedvcmr;

import android.app.Service;
import android.content.Intent;
import android.os.Debug;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import com.fedvcmr.core.SearchEngine;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * VCMR E2E Accuracy, Latency, Memory & Battery Benchmark.
 *
 * Trigger: adb shell am startservice -a com.fedvcmr.START_BENCHMARK -p com.fedvcmr
 * Output CSV: /sdcard/fedvcmr/benchmark_metrics.csv
 */
public class BenchmarkService extends Service {

    private static final String TAG = "BENCH_RESULT";
    private static final String QUERY_FILE = "/sdcard/fedvcmr/benchmark_queries.json";
    private static final String CSV_FILE   = "/sdcard/fedvcmr/benchmark_metrics.csv";

    private SearchEngine searchEngine;
    private PowerManager.WakeLock wakeLock;
    private static final String CHANNEL_ID = "BenchmarkChannel";

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        android.app.Notification notification = new android.app.Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("VCMR Benchmark")
                .setContentText("Running 10,000 queries...")
                .setSmallIcon(android.R.drawable.ic_menu_info_details)
                .build();
        startForeground(1, notification, android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);

        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, "fedvcmr:BenchmarkLock");
    }

    private void createNotificationChannel() {
        android.app.NotificationChannel channel = new android.app.NotificationChannel(
                CHANNEL_ID, "Benchmark Service", android.app.NotificationManager.IMPORTANCE_LOW);
        android.app.NotificationManager manager = getSystemService(android.app.NotificationManager.class);
        manager.createNotificationChannel(channel);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "com.fedvcmr.START_BENCHMARK".equals(intent.getAction())) {
            new Thread(this::runFullBenchmark).start();
        }
        return START_STICKY;
    }

    // Returns current charge in mAh. BATTERY_PROPERTY_CHARGE_COUNTER is not affected by dumpsys override.
    private int getChargeCounterMah() {
        android.os.BatteryManager bm = (android.os.BatteryManager) getSystemService(BATTERY_SERVICE);
        int uah = bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        return uah / 1000; // µAh → mAh
    }

    private long getMemoryUsageMB() {
        Runtime rt = Runtime.getRuntime();
        long javaHeap = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
        long nativeHeap = Debug.getNativeHeapAllocatedSize() / (1024 * 1024);
        return javaHeap + nativeHeap;
    }

    private void runFullBenchmark() {
        wakeLock.acquire();
        Log.i(TAG, "--- BENCHMARK START ---");

        PrintWriter csvWriter = null;
        long benchmarkStartMs = System.currentTimeMillis();

        try {
            if (searchEngine == null) searchEngine = new SearchEngine(this);

            int startChargeMah = getChargeCounterMah();
            Log.i(TAG, "Initial Charge Counter: " + startChargeMah + " mAh");

            File file = new File(QUERY_FILE);
            if (!file.exists()) {
                Log.e(TAG, "Benchmark file not found: " + QUERY_FILE);
                return;
            }

            JSONArray queries = new JSONArray(new String(VideoIngestor.readFile(file)));
            int total = queries.length();
            Log.i(TAG, "Loaded " + total + " queries for benchmark.");

            // CSV header
            csvWriter = new PrintWriter(new FileWriter(CSV_FILE, false));
            csvWriter.println("query_num,timestamp_ms,elapsed_sec,latency_ms,r1_pct,r5_pct,charge_mah,drain_mah,memory_mb,peak_ram_mb");

            int top1Hits = 0;
            int top5Hits = 0;
            long totalLatency = 0;
            long peakMemoryMB = 0;

            for (int i = 0; i < total; i++) {
                JSONObject qObj = queries.getJSONObject(i);
                String queryText = qObj.getString("query");
                String gtVideo   = qObj.getString("video_id");

                long startTime = System.currentTimeMillis();
                List<SearchEngine.Hit> results = searchEngine.search(queryText);
                long latency = System.currentTimeMillis() - startTime;
                totalLatency += latency;

                if (!results.isEmpty()) {
                    for (int k = 0; k < Math.min(5, results.size()); k++) {
                        if (results.get(k).videoId.equals(gtVideo)) {
                            if (k == 0) {
                                top1Hits++;
                                SearchEngine.Hit hit = results.get(0);
                                saveVerificationFrame(gtVideo, (hit.tStart + hit.tEnd) / 2.0, queryText);
                            }
                            top5Hits++;
                            break;
                        }
                    }
                }

                long memMB = getMemoryUsageMB();
                if (memMB > peakMemoryMB) peakMemoryMB = memMB;

                // Write CSV row every query
                long nowMs = System.currentTimeMillis();
                float elapsedSec = (nowMs - benchmarkStartMs) / 1000f;
                float r1 = i > 0 ? (top1Hits / (float)(i + 1)) * 100 : 0;
                float r5 = i > 0 ? (top5Hits / (float)(i + 1)) * 100 : 0;
                int curChargeMah = (i % 50 == 0) ? getChargeCounterMah() : -1; // read every 50 queries to save I/O
                int drainMah = (curChargeMah >= 0) ? (startChargeMah - curChargeMah) : -1;
                csvWriter.printf("%d,%d,%.1f,%d,%.2f,%.2f,%d,%d,%d,%d%n",
                        i + 1, nowMs, elapsedSec, latency,
                        r1, r5,
                        curChargeMah, drainMah,
                        memMB, peakMemoryMB);

                // Log progress every 10 queries
                if (i > 0 && i % 10 == 0) {
                    float avgLat = totalLatency / (float)(i + 1);
                    int curMah = getChargeCounterMah();
                    Log.i(TAG, String.format("Progress: %d/%d | R@1: %.2f%% | R@5: %.2f%% | AvgLat: %.0fms | Charge: %dmAh | Drain: %dmAh | Mem: %dMB | PeakRAM: %dMB",
                            i, total, r1, r5, avgLat, curMah, startChargeMah - curMah, memMB, peakMemoryMB));
                }
            }

            csvWriter.flush();

            int endChargeMah = getChargeCounterMah();
            float totalDurationMin = (System.currentTimeMillis() - benchmarkStartMs) / 60000f;

            Log.i(TAG, "--- BENCHMARK COMPLETE ---");
            Log.i(TAG, String.format(
                    "Final Metrics [ActivityNet 24h]: Queries=%d | R@1=%.2f%% | R@5=%.2f%% | AvgLat=%.2fms | BatteryDrain=%dmAh | DrainRate=%.2fmAh/hr | Duration=%.1fmin | PeakRAM=%dMB",
                    total,
                    (top1Hits / (float) total) * 100,
                    (top5Hits / (float) total) * 100,
                    (totalLatency / (float) total),
                    startChargeMah - endChargeMah,
                    (startChargeMah - endChargeMah) / (totalDurationMin / 60f),
                    totalDurationMin,
                    peakMemoryMB));
            Log.i(TAG, "CSV saved to: " + CSV_FILE);

        } catch (Exception e) {
            Log.e(TAG, "Benchmark execution failed", e);
        } finally {
            if (csvWriter != null) csvWriter.close();
            if (wakeLock.isHeld()) wakeLock.release();
            stopSelf();
        }
    }

    private void saveVerificationFrame(String videoId, double centerTimeSec, String query) {
        File videoFile = new File("/sdcard/fedvcmr/user_videos/" + videoId + ".mp4");
        if (!videoFile.exists()) videoFile = new File("/sdcard/fedvcmr/ActivityNet/videos/" + videoId + ".mp4");
        if (!videoFile.exists()) videoFile = new File("/sdcard/fedvcmr/ActivityNet/videos/videos/" + videoId + ".mp4");
        if (!videoFile.exists()) return;

        try {
            android.media.MediaMetadataRetriever mmr = new android.media.MediaMetadataRetriever();
            mmr.setDataSource(videoFile.getAbsolutePath());
            android.graphics.Bitmap frame = mmr.getFrameAtTime(
                    (long)(centerTimeSec * 1_000_000),
                    android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
            if (frame != null) {
                File outDir = new File("/sdcard/fedvcmr/verification");
                outDir.mkdirs();
                String cleanQuery = query.replaceAll("[^a-zA-Z0-9]", "_");
                if (cleanQuery.length() > 30) cleanQuery = cleanQuery.substring(0, 30);
                File outFile = new File(outDir, videoId + "_" + cleanQuery + ".jpg");
                try (FileOutputStream out = new FileOutputStream(outFile)) {
                    frame.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out);
                }
                frame.recycle();
                Log.d(TAG, "Saved verification frame: " + outFile.getName());
            }
            mmr.release();
        } catch (Exception e) {
            Log.e(TAG, "Frame save failed", e);
        }
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }
}
