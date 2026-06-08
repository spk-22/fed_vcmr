package com.fedvcmr;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

/**
 * Minimal launcher activity. Opens once to "activate" the app,
 * then closes immediately. Required by Android 13+ for receivers to work.
 */
public class InitActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i("FEDVCMR", "App activated. Receivers are now live.");
        finish();
    }
}
