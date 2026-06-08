package com.fedvcmr;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

/**
 * VCMR 3-stage search pipeline:
 *   Stage 1: Cosine similarity on pre-pushed embeddings (CPU)
 *   Stage 2: DGSE reranking via dgse.tflite (NPU/GPU)
 *   Stage 3: Temporal grounding via temporal_grounding.tflite (NPU/GPU)
 *
 * Trigger: adb shell am startservice -n com.fedvcmr/.SearchService --es query "jumping into pool"
 */
public class SearchService extends Service {

    private static final String TAG = "VCMR_RESULT";

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String query = intent.getStringExtra("query");
        Log.i(TAG, "Search requested: " + query);

        // TODO: Stage 1 - Cosine similarity
        // TODO: Stage 2 - DGSE rerank
        // TODO: Stage 3 - Temporal grounding

        stopSelf();
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
