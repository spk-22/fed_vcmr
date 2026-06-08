package com.fedvcmr.ui;

import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.fedvcmr.R;
import com.fedvcmr.VideoIngestor;
import com.fedvcmr.core.SearchEngine;
import com.google.android.material.snackbar.Snackbar;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private SearchEngine searchEngine;
    private ResultsAdapter adapter;
    private ExecutorService executor = Executors.newSingleThreadExecutor();

    private EditText searchView;
    private RecyclerView resultsRecyclerView;
    private ProgressBar loadingProgress;
    private TextView emptyStateText;
    private TextView indexCountText;
    private TextView latencyText;

    private final ActivityResultLauncher<String> pickerLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(),
            uri -> { if (uri != null) startIngest(uri); }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        searchView = findViewById(R.id.searchView);
        resultsRecyclerView = findViewById(R.id.resultsRecyclerView);
        loadingProgress = findViewById(R.id.loadingProgress);
        emptyStateText = findViewById(R.id.emptyStateText);
        indexCountText = findViewById(R.id.indexCountText);
        latencyText = findViewById(R.id.latencyText);

        resultsRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        adapter = new ResultsAdapter(this, new ArrayList<>());
        resultsRecyclerView.setAdapter(adapter);

        findViewById(R.id.ingestButton).setOnClickListener(v -> pickerLauncher.launch("video/*"));
        findViewById(R.id.clearButton).setOnClickListener(v -> clearIndex());

        searchView.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                submitSearch(searchView.getText().toString());
                return true;
            }
            return false;
        });

        initEngine();
    }

    private void initEngine() {
        executor.execute(() -> {
            try {
                searchEngine = new SearchEngine(this);
                runOnUiThread(() -> {
                    updateIndexCount();
                    Log.d(TAG, "SearchEngine ready");
                });
            } catch (Exception e) {
                Log.e(TAG, "Engine init failed", e);
                runOnUiThread(() -> showError("Engine init failed: " + e.getMessage()));
            }
        });
    }

    private void submitSearch(String query) {
        if (TextUtils.isEmpty(query)) return;
        hideAll();
        resultsRecyclerView.setVisibility(View.GONE);
        loadingProgress.setVisibility(View.VISIBLE);

        executor.execute(() -> {
            try {
                if (searchEngine == null) {
                    runOnUiThread(() -> {
                        loadingProgress.setVisibility(View.GONE);
                        showError("Engine unavailable");
                    });
                    return;
                }

                long t0 = System.currentTimeMillis();
                List<SearchEngine.Hit> results = searchEngine.search(query);
                long ms = System.currentTimeMillis() - t0;
                Log.i(TAG, "Search UI: query=" + query + " results=" + results.size() + " ms=" + ms);

                runOnUiThread(() -> {
                    loadingProgress.setVisibility(View.GONE);
                    if (results.isEmpty()) {
                        Log.i(TAG, "Search UI: Showing empty state");
                        showEmpty("No results for \"" + query + "\"");
                    } else {
                        Log.i(TAG, "Search UI: Submitting to adapter: " + results.size());
                        adapter.submitList(results);
                        resultsRecyclerView.setVisibility(View.VISIBLE);
                        latencyText.setText("Latency: " + ms + " ms");
                        latencyText.setVisibility(View.VISIBLE);
                    }
                });
            } catch (Exception e) {
                Log.e(TAG, "Search failed: " + e.getMessage(), e);
                runOnUiThread(() -> {
                    loadingProgress.setVisibility(View.GONE);
                    showError("Search failed: " + e.getMessage());
                });
            }
        });
    }

    private void startIngest(Uri uri) {
        runOnUiThread(() -> { hideAll(); loadingProgress.setVisibility(View.VISIBLE); });

        executor.execute(() -> VideoIngestor.ingest(this, uri, searchEngine, new VideoIngestor.Callback() {
            @Override
            public void onProgress(String message) {
                runOnUiThread(() -> showEmpty(message));
            }

            @Override
            public void onSuccess(String videoId, int chunksAdded, int totalUserChunks) {
                runOnUiThread(() -> {
                    loadingProgress.setVisibility(View.GONE);
                    showEmpty("Enter a search query");
                    updateIndexCount();
                    Snackbar.make(searchView,
                        "Indexed: " + videoId + " (32 frames sampled)",
                        Snackbar.LENGTH_LONG).show();
                    Log.i(TAG, "Ingest complete: " + videoId + " total videos=" + totalUserChunks);
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    loadingProgress.setVisibility(View.GONE);
                    showError("Ingest failed: " + error);
                });
            }
        }));
    }

    private void clearIndex() {
        executor.execute(() -> {
            try {
                File dir = new File(VideoIngestor.DATA_DIR);
                deleteRecursive(dir);
                searchEngine = new SearchEngine(this);
                runOnUiThread(() -> {
                    adapter.submitList(new ArrayList<>());
                    updateIndexCount();
                    Toast.makeText(this, "Index cleared", Toast.LENGTH_SHORT).show();
                });
            } catch (Exception e) {
                runOnUiThread(() -> showError("Reset failed"));
            }
        });
    }

    private void deleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory()) {
            File[] files = fileOrDirectory.listFiles();
            if (files != null) for (File child : files) deleteRecursive(child);
        }
        fileOrDirectory.delete();
    }

    private void updateIndexCount() {
        if (searchEngine == null) return;
        int yours = searchEngine.getVideoCount();
        String label = yours == 0 ? "No videos indexed yet" : yours + " video" + (yours == 1 ? "" : "s") + " indexed";
        indexCountText.setText(label);
        indexCountText.setVisibility(View.VISIBLE);
    }

    private void showEmpty(String msg) {
        emptyStateText.setText(msg);
        emptyStateText.setVisibility(View.VISIBLE);
    }

    private void showError(String msg) {
        Snackbar.make(searchView, msg, Snackbar.LENGTH_LONG).show();
    }

    private void hideAll() {
        resultsRecyclerView.setVisibility(View.GONE);
        loadingProgress.setVisibility(View.GONE);
        latencyText.setVisibility(View.GONE);
        emptyStateText.setVisibility(View.GONE);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
        if (searchEngine != null) searchEngine.close();
    }
}
