package com.fedvcmr.ui;

import android.content.Context;
import android.graphics.Color;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ProgressBar;
import android.widget.TextView;

import android.net.Uri;

import androidx.annotation.NonNull;
import androidx.media3.common.MediaItem;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.ui.PlayerView;
import androidx.recyclerview.widget.RecyclerView;

import com.fedvcmr.R;
import com.fedvcmr.core.SearchEngine;

import java.io.File;
import java.util.List;

/**
 * RecyclerView adapter for search results with inline ExoPlayer.
 *
 * Each row shows source badge ("Demo" / "Your video"), video ID, moment
 * timing, relevance score bar, and inline video player clipped to the
 * detected moment bounds.
 */
public class ResultsAdapter extends RecyclerView.Adapter<ResultsAdapter.ResultViewHolder> {

    private static final String TAG = "ResultsAdapter";

    private Context context;
    private List<SearchEngine.Hit> results;

    public ResultsAdapter(Context context, List<SearchEngine.Hit> results) {
        this.context = context;
        this.results = results;
    }

    public void submitList(List<SearchEngine.Hit> newResults) {
        this.results = newResults;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ResultViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(context).inflate(R.layout.item_result, parent, false);
        return new ResultViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull ResultViewHolder holder, int position) {
        holder.bind(results.get(position));
    }

    @Override
    public int getItemCount() { return results.size(); }

    @Override
    public void onViewRecycled(@NonNull ResultViewHolder holder) {
        super.onViewRecycled(holder);
        holder.releasePlayer();
    }

    public static class ResultViewHolder extends RecyclerView.ViewHolder {

        private PlayerView playerView;
        private TextView   videoIdText;
        private TextView   momentText;
        private ProgressBar scoreBar;
        private TextView   scoreText;
        private TextView   sourceBadge;
        private ExoPlayer  player;

        public ResultViewHolder(@NonNull View itemView) {
            super(itemView);
            playerView  = itemView.findViewById(R.id.playerView);
            videoIdText = itemView.findViewById(R.id.videoIdText);
            momentText  = itemView.findViewById(R.id.momentText);
            scoreBar    = itemView.findViewById(R.id.scoreBar);
            scoreText   = itemView.findViewById(R.id.scoreText);
            sourceBadge = itemView.findViewById(R.id.sourceBadge);
        }

        public void bind(SearchEngine.Hit hit) {
            videoIdText.setText(hit.videoId);
            momentText.setText(String.format("Moment: %.1fs - %.1fs", hit.tStart, hit.tEnd));
            scoreText.setText(String.format("%.3f", hit.score));
            scoreBar.setProgress(Math.min(100, (int)(hit.score * 100)));

            // Source badge + video directory
            String videoDir;
            if (hit.fused) {
                sourceBadge.setText("Fused");
                sourceBadge.setBackgroundColor(Color.parseColor("#E65100")); // deep orange
            } else {
                sourceBadge.setText("Your video");
                sourceBadge.setBackgroundColor(Color.parseColor("#388E3C")); // green
            }
            videoDir = "/sdcard/fedvcmr/user_videos";
            String videoPath = videoDir + "/" + hit.videoId + ".mp4";
            setupPlayer(videoPath, hit.tStart, hit.tEnd);

            itemView.setContentDescription(String.format(
                "%s result: %s at %.1fs-%.1fs score %.3f",
                hit.fused ? "Fused" : "Your video",
                hit.videoId, hit.tStart, hit.tEnd, hit.score
            ));
        }

        private void setupPlayer(String videoPath, double tStart, double tEnd) {
            try {
                if (player == null) {
                    player = new ExoPlayer.Builder(itemView.getContext()).build();
                    playerView.setPlayer(player);
                }

                File videoFile = new File(videoPath);
                Log.d(TAG, "Checking video file: " + videoPath + " exists=" + videoFile.exists() + " readable=" + videoFile.canRead());
                if (!videoFile.exists()) {
                    Log.w(TAG, "Video file not found: " + videoPath);
                    return;
                }

                long momentStartMs = (long)(tStart * 1000);
                long momentEndMs   = (long)(tEnd   * 1000);

                // Use file:// URI so ExoPlayer resolves the path correctly
                Uri videoUri = Uri.fromFile(videoFile);
                MediaItem mediaItem = new MediaItem.Builder()
                    .setUri(videoUri)
                    .setClippingConfiguration(new MediaItem.ClippingConfiguration.Builder()
                        .setStartPositionMs(momentStartMs)
                        .setEndPositionMs(momentEndMs)
                        .build())
                    .build();

                player.setMediaItem(mediaItem);
                player.prepare();
                player.setPlayWhenReady(true);   // auto-play moment on scroll into view

                Log.d(TAG, "Player ready: " + videoPath + " [" + tStart + "s-" + tEnd + "s]");
            } catch (Exception e) {
                Log.e(TAG, "Player setup failed: " + e.getMessage(), e);
            }
        }

        public void releasePlayer() {
            if (player != null) {
                player.release();
                player = null;
            }
        }
    }
}
