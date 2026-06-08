package com.fedvcmr;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.SystemClock;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * On-device FL training using PyTorch Mobile (native ARM BLAS).
 * All matrix ops run through LibTorch — 100x faster than Java loops.
 *
 * Trigger:
 *   adb shell am broadcast -n com.fedvcmr/.FLReceiver --ei epochs 10 --ei batch_size 16 --ei max_samples 500
 */
public class FLReceiver extends BroadcastReceiver {

    private static final String TAG = "FL_RESULT";
    private static final String DATA_DIR = "/sdcard/fedvcmr";
    private static final int DIM = 512;
    private static final float LR = 1e-5f;
    private static final float TEMP = 0.07f;
    private static final float MU = 0.01f;

    private static final int K_HARD = 4;
    private static float[] cachedVectorMap;

    @Override
    public void onReceive(Context context, Intent intent) {
        int epochs = intent.getIntExtra("epochs", 3);
        int batchSize = intent.getIntExtra("batch_size", 16);
        int maxSamples = intent.getIntExtra("max_samples", 100);
        final Context ctx = context.getApplicationContext();
        new Thread(() -> {
            try {
                ensureDataLoaded();
                train(ctx, epochs, batchSize, maxSamples);
            } catch (Exception e) {
                Log.e(TAG, "Mining init failed", e);
            }
        }).start();
    }

    private void ensureDataLoaded() throws Exception {
        if (cachedVectorMap == null) {
            cachedVectorMap = NpyReader.readNpy(DATA_DIR + "/vector_map.npy");
        }
    }

    private void train(Context ctx, int epochs, int batchSize, int maxSamples) {
        try {
            long startTime = SystemClock.elapsedRealtimeNanos();
            Log.i(TAG, "FL training starting (PyTorch Mobile): epochs=" + epochs
                    + " batch_size=" + batchSize + " max_samples=" + maxSamples);

            // 1. Load TorchScript trainer module from assets
            String trainerPath = assetFilePath(ctx, "models/fl_trainer.pt");
            Module trainer = Module.load(trainerPath);
            Log.i(TAG, "TorchScript trainer loaded");

            // 2. Load global weights (512x512 float32)
            float[] textWeightsData = NpyReader.readNpy(DATA_DIR + "/fl_text_weights.npy");
            float[] visionWeightsData = NpyReader.readNpy(DATA_DIR + "/fl_vision_weights.npy");

            // Keep copies for proximal regularization
            float[] textWeightsGlobalData = textWeightsData.clone();
            float[] visionWeightsGlobalData = visionWeightsData.clone();

            // 3. Load text features
            float[] allTextData = NpyReader.readNpy(DATA_DIR + "/fl_text_features.npy");
            int totalSamples = allTextData.length / DIM;

            // 4. Load vision features (mean-pooled, normalized)
            java.io.FileInputStream fis = new java.io.FileInputStream(DATA_DIR + "/chunk_index.json");
            byte[] jsonBytes = fis.readAllBytes();
            fis.close();
            org.json.JSONArray chunkIndex = new org.json.JSONArray(new String(jsonBytes));

            int maxToLoad = Math.min(Math.min(totalSamples, chunkIndex.length()), maxSamples);
            List<float[]> validText = new ArrayList<>();
            List<float[]> validVision = new ArrayList<>();

            long loadStart = SystemClock.elapsedRealtimeNanos();
            for (int i = 0; i < maxToLoad; i++) {
                String chunkId = chunkIndex.getJSONObject(i).getString("chunk_id");
                try {
                    float[] frames = NpyReader.readNpy(DATA_DIR + "/features/" + chunkId + ".npy");
                    float[] pooled = new float[DIM];
                    // Mean-pool 8 frames
                    for (int d = 0; d < DIM; d++) {
                        float sum = 0;
                        for (int f = 0; f < 8; f++) sum += frames[f * DIM + d];
                        pooled[d] = sum / 8.0f;
                    }
                    // L2 normalize
                    float norm = 0;
                    for (int d = 0; d < DIM; d++) norm += pooled[d] * pooled[d];
                    norm = (float) Math.sqrt(norm) + 1e-8f;
                    for (int d = 0; d < DIM; d++) pooled[d] /= norm;

                    validVision.add(pooled);
                    float[] text = new float[DIM];
                    System.arraycopy(allTextData, i * DIM, text, 0, DIM);
                    validText.add(text);
                } catch (Exception e) {
                    // Skip missing chunks
                }
            }
            int numSamples = validVision.size();
            float[] finalVisionData = new float[numSamples * DIM];
            float[] finalTextData = new float[numSamples * DIM];
            for (int i = 0; i < numSamples; i++) {
                System.arraycopy(validVision.get(i), 0, finalVisionData, i * DIM, DIM);
                System.arraycopy(validText.get(i), 0, finalTextData, i * DIM, DIM);
            }
            
            long loadTime = (SystemClock.elapsedRealtimeNanos() - loadStart) / 1_000_000;
            Log.i(TAG, "Data loaded: " + numSamples + " samples in " + loadTime + "ms");

            if (numSamples < 2) {
                Log.e(TAG, "Not enough samples for training: " + numSamples);
                return;
            }

            // 5. Training loop using PyTorch Mobile
            Tensor textW = Tensor.fromBlob(textWeightsData, new long[]{DIM, DIM});
            Tensor visionW = Tensor.fromBlob(visionWeightsData, new long[]{DIM, DIM});
            Tensor textWGlobal = Tensor.fromBlob(textWeightsGlobalData, new long[]{DIM, DIM});
            Tensor visionWGlobal = Tensor.fromBlob(visionWeightsGlobalData, new long[]{DIM, DIM});

            // NC5: Moment Codebook (16 prototypes x 512 dim)
            int M = 16;
            float[] codebookData = new float[M * DIM];
            // Initialize with some variation (random-like)
            for (int i = 0; i < codebookData.length; i++) codebookData[i] = (float) (Math.random() - 0.5) * 0.1f;
            Tensor codebook = Tensor.fromBlob(codebookData, new long[]{M, DIM});
            float[] codebookInitial = codebookData.clone();

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) indices.add(i);

            float totalLoss = 0;
            int totalBatches = 0;

            for (int epoch = 0; epoch < epochs; epoch++) {
                long epochStart = SystemClock.elapsedRealtimeNanos();
                Collections.shuffle(indices);
                float epochLoss = 0;
                int epochBatches = 0;

                for (int b = 0; b < numSamples; b += batchSize) {
                    int end = Math.min(b + batchSize, numSamples);
                    int bs = end - b;
                    if (bs < 2) continue;

                    // Gather batch data
                    float[] batchText = new float[bs * DIM];
                    float[] batchVision = new float[bs * DIM];
                    float[] batchHardNeg = new float[bs * K_HARD * DIM];

                    for (int i = 0; i < bs; i++) {
                        int idx = indices.get(b + i);
                        System.arraycopy(finalTextData, idx * DIM, batchText, i * DIM, DIM);
                        System.arraycopy(finalVisionData, idx * DIM, batchVision, i * DIM, DIM);

                        // NC3: Mine hard negatives for this sample
                        float[] textEmb = new float[DIM];
                        System.arraycopy(finalTextData, idx * DIM, textEmb, 0, DIM);
                        int[] hardIndices = findHardNegatives(textEmb, idx, K_HARD, finalVisionData);
                        for (int k = 0; k < K_HARD; k++) {
                            int hIdx = hardIndices[k];
                            System.arraycopy(finalVisionData, hIdx * DIM, batchHardNeg, (i * K_HARD + k) * DIM, DIM);
                        }
                    }

                    Tensor textBatch = Tensor.fromBlob(batchText, new long[]{bs, DIM});
                    Tensor visionBatch = Tensor.fromBlob(batchVision, new long[]{bs, DIM});
                    Tensor hardNegBatch = Tensor.fromBlob(batchHardNeg, new long[]{bs * K_HARD, DIM});

                    // Call train_step with hard negatives and codebook
                    IValue result = trainer.runMethod("train_step",
                            IValue.from(textBatch),
                            IValue.from(visionBatch),
                            IValue.from(hardNegBatch),
                            IValue.from(textW),
                            IValue.from(visionW),
                            IValue.from(codebook),
                            IValue.from(textWGlobal),
                            IValue.from(visionWGlobal),
                            IValue.from(LR),
                            IValue.from(TEMP),
                            IValue.from(MU));

                    // Unpack result tuple: (loss, text_W_new, vision_W_new, codebook_new)
                    IValue[] outputs = result.toTuple();
                    float loss = outputs[0].toTensor().getDataAsFloatArray()[0];
                    textW = outputs[1].toTensor();
                    visionW = outputs[2].toTensor();
                    codebook = outputs[3].toTensor();

                    epochLoss += loss;
                    epochBatches++;
                }

                long epochTime = (SystemClock.elapsedRealtimeNanos() - epochStart) / 1_000_000;
                float avgLoss = epochBatches > 0 ? epochLoss / epochBatches : 0;
                totalLoss += epochLoss;
                totalBatches += epochBatches;
                Log.i(TAG, "Epoch " + (epoch + 1) + "/" + epochs
                        + " loss=" + String.format("%.4f", avgLoss)
                        + " time=" + epochTime + "ms");
            }

            // 6. Save updated weights and codebook
            float[] updatedText = textW.getDataAsFloatArray();
            float[] updatedVision = visionW.getDataAsFloatArray();
            float[] updatedCodebook = codebook.getDataAsFloatArray();

            saveNpy(updatedText, DIM, DIM, DATA_DIR + "/fl_text_weights_updated.npy");
            saveNpy(updatedVision, DIM, DIM, DATA_DIR + "/fl_vision_weights_updated.npy");
            saveNpy(updatedCodebook, M, DIM, DATA_DIR + "/fl_moment_codebook_updated.npy");

            long elapsed = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000;
            float avgLoss = totalBatches > 0 ? totalLoss / totalBatches : 0;

            // Compute deltas
            float textDelta = 0, visionDelta = 0, cbDelta = 0;
            for (int i = 0; i < DIM * DIM; i++) {
                float dt = updatedText[i] - textWeightsGlobalData[i];
                float dv = updatedVision[i] - visionWeightsGlobalData[i];
                textDelta += dt * dt;
                visionDelta += dv * dv;
            }
            for (int i = 0; i < M * DIM; i++) {
                float dc = updatedCodebook[i] - codebookInitial[i];
                cbDelta += dc * dc;
            }
            textDelta = (float) Math.sqrt(textDelta);
            visionDelta = (float) Math.sqrt(visionDelta);
            cbDelta = (float) Math.sqrt(cbDelta);

            Log.i(TAG, String.format(
                "FL complete | NC5 Codebook Active | avg_loss=%.4f | text_d=%.6f | vision_d=%.6f | cb_d=%.6f | duration=%dms",
                avgLoss, textDelta, visionDelta, cbDelta, elapsed));

            trainer.destroy();

        } catch (Exception e) {
            Log.e(TAG, "FL training failed: " + e.getMessage(), e);
        }
    }

    private int[] findHardNegatives(float[] query, int excludeIdx, int k, float[] localPool) {
        int numChunks = localPool.length / DIM;
        float[] scores = new float[numChunks];
        for (int i = 0; i < numChunks; i++) {
            if (i == excludeIdx) {
                scores[i] = -1.0f;
                continue;
            }
            float dot = 0;
            int offset = i * DIM;
            for (int j = 0; j < DIM; j++) {
                dot += query[j] * localPool[offset + j];
            }
            scores[i] = dot;
        }

        int[] indices = new int[numChunks];
        for (int i = 0; i < numChunks; i++) indices[i] = i;

        // Partial sort for top-k
        int actualK = Math.min(k, numChunks - 1);
        for (int i = 0; i < actualK; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < numChunks; j++) {
                if (scores[indices[j]] > scores[indices[maxIdx]]) maxIdx = j;
            }
            int tmp = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = tmp;
        }
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = indices[i % numChunks]; // fallback if k > numChunks
        }
        return out;
    }

    private static String assetFilePath(Context context, String assetName) throws Exception {
        File file = new File(context.getFilesDir(), assetName.replace("/", "_"));
        // Always overwrite to ensure latest TorchScript model from assets is used
        InputStream is = context.getAssets().open(assetName);
        FileOutputStream fos = new FileOutputStream(file);
        byte[] buffer = new byte[4096];
        int read;
        while ((read = is.read(buffer)) != -1) fos.write(buffer, 0, read);
        fos.close();
        is.close();
        return file.getAbsolutePath();
    }

    /** Save a 2D float array as .npy (float32, C-order) */
    private void saveNpy(float[] data, int rows, int cols, String path) throws Exception {
        String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + rows + ", " + cols + "), }";
        int headerLen = header.length() + 1;
        int padded = ((headerLen + 10 + 63) / 64) * 64;
        int padding = padded - 10 - headerLen;
        StringBuilder sb = new StringBuilder(header);
        for (int i = 0; i < padding; i++) sb.append(' ');
        sb.append('\n');
        String paddedHeader = sb.toString();

        FileOutputStream fos = new FileOutputStream(path);
        fos.write(new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'});
        fos.write(new byte[]{1, 0});
        int hLen = paddedHeader.length();
        fos.write(new byte[]{(byte) (hLen & 0xFF), (byte) ((hLen >> 8) & 0xFF)});
        fos.write(paddedHeader.getBytes());
        ByteBuffer buf = ByteBuffer.allocate(data.length * 4);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) buf.putFloat(v);
        fos.write(buf.array());
        fos.close();
    }
}
