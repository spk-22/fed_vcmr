package com.fedvcmr;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

/**
 * Federated Learning client service.
 * Loads projection heads, trains locally, sends weights to PC server via gRPC.
 *
 * Trigger: adb shell am startservice -n com.fedvcmr/.FLService --ei rounds 1
 */
public class FLService extends Service {

    private static final String TAG = "FL_RESULT";

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        int rounds = intent.getIntExtra("rounds", 1);
        Log.i(TAG, "FL training requested: " + rounds + " rounds");

        // TODO: Load projection heads
        // TODO: Local training on device features
        // TODO: Send weights via gRPC to localhost:8080

        stopSelf();
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
