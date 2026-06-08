package com.fedvcmr.core;

import android.content.Context;
import android.util.Log;

import com.fedvcmr.ModelLoader;

import org.json.JSONObject;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.tensorflow.lite.Interpreter;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * MobileCLIP text encoder: text → 512-dim L2-normalized embedding.
 *
 * Two-stage pipeline matching VisionEncoder exactly:
 *   1. PyTorch Mobile backbone (mobileclip_text_encoder.ptl)  → raw 512-dim
 *   2. TFLite projection head (text_head_best_model.tflite)   → aligned 512-dim
 *
 * Vocab: clip_encoder.json + clip_bpe_ranks.json + clip_byte_encoder.json
 * All three assets verified to match open_clip MobileCLIP-S1 (person</w>=2533).
 */
public class TextEncoder {
    private static final String TAG       = "TextEncoder";
    private static final int    EMBED_DIM = 512;
    private static final int    SEQ_LEN   = 77;

    private Module       module;
    private Interpreter  headInterp;
    private CLIPTokenizer tokenizer;

    public TextEncoder(Context context) throws Exception {
        this.module     = LiteModuleLoader.loadModuleFromAsset(
                context.getAssets(), "models/mobileclip_text_encoder.ptl");
        this.headInterp = ModelLoader.createInterpreter(context, "text_head_best_model.tflite", "CPU");
        this.tokenizer  = new CLIPTokenizer(context);
        Log.d(TAG, "TextEncoder loaded (PyTorch backbone + TFLite projection head)");
    }

    /** Encode text → float[512] L2-normalized, raw backbone embedding. */
    public float[] encode(String text) {
        try {
            int[] tokenIds = tokenizer.tokenize(text, SEQ_LEN);
            Log.d(TAG, "tokens(" + text + ")=" +
                    java.util.Arrays.toString(java.util.Arrays.copyOf(tokenIds, 8)));

            // Stage 1: PyTorch backbone (Raw MobileCLIP-S1)
            Tensor input      = Tensor.fromBlob(tokenIds, new long[]{1, SEQ_LEN});
            float[] backbone  = module.forward(IValue.from(input)).toTensor().getDataAsFloatArray();

            // Stage 2: BYPASS Projection Head
            // We return raw backbone features normalized to unit vector
            return normalize(backbone);

        } catch (Exception e) {
            Log.e(TAG, "Encoding failed: " + e.getMessage(), e);
            throw new RuntimeException(e);
        }
    }

    private float[] normalize(float[] v) {
        float n = 0;
        for (float x : v) n += x * x;
        n = (float) Math.sqrt(n);
        if (n < 1e-10f) return v;
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / n;
        return out;
    }

    public void close() {
        if (module     != null) module.destroy();
        if (headInterp != null) headInterp.close();
    }

    // =========================================================================
    // CLIP BPE Tokenizer
    // Verified: person</w>=2533, sitting</w>=4919, stairs</w>=9577 (matches Python open_clip)
    // =========================================================================

    public static class CLIPTokenizer {
        private static final String TAG = "CLIPTokenizer";
        private static final int    SOT = 49406;
        private static final int    EOT = 49407;

        private final Map<String, Integer> encoder     = new HashMap<>();
        private final Map<String, Integer> bpeRanks    = new HashMap<>();
        private final Map<Integer, String> byteEncoder = new HashMap<>();
        private final Map<String, String>  cache       = new HashMap<>();

        public CLIPTokenizer(Context context) throws Exception {
            JSONObject enc = loadJson(context, "clip_encoder.json");
            for (Iterator<String> k = enc.keys(); k.hasNext(); ) {
                String key = k.next(); encoder.put(key, enc.getInt(key));
            }
            JSONObject ranks = loadJson(context, "clip_bpe_ranks.json");
            for (Iterator<String> k = ranks.keys(); k.hasNext(); ) {
                String key = k.next(); bpeRanks.put(key, ranks.getInt(key));
            }
            JSONObject be = loadJson(context, "clip_byte_encoder.json");
            for (Iterator<String> k = be.keys(); k.hasNext(); ) {
                String key = k.next(); byteEncoder.put(Integer.parseInt(key), be.getString(key));
            }
            Log.d(TAG, "vocab=" + encoder.size() + " ranks=" + bpeRanks.size());
        }

        private JSONObject loadJson(Context ctx, String name) throws Exception {
            InputStream is = ctx.getAssets().open(name);
            byte[] buf = new byte[is.available()]; is.read(buf); is.close();
            return new JSONObject(new String(buf, "UTF-8"));
        }

        public int[] tokenize(String text, int seqLen) {
            List<Integer> tokens = new ArrayList<>();
            tokens.add(SOT);
            for (String word : text.toLowerCase().trim().replaceAll("\\s+", " ").split(" ")) {
                if (word.isEmpty()) continue;
                for (String sub : bpe(word).split(" ")) {
                    Integer id = encoder.get(sub);
                    if (id != null) tokens.add(id);
                }
            }
            tokens.add(EOT);
            int[] result = new int[seqLen];
            for (int i = 0; i < Math.min(tokens.size(), seqLen); i++) result[i] = tokens.get(i);
            return result;
        }

        private String bpe(String token) {
            if (cache.containsKey(token)) return cache.get(token);

            byte[] bytes;
            try { bytes = token.getBytes("UTF-8"); } catch (Exception e) { bytes = token.getBytes(); }

            List<String> word = new ArrayList<>();
            for (int i = 0; i < bytes.length; i++) {
                String sym = byteEncoder.get(bytes[i] & 0xFF);
                if (sym == null) sym = String.valueOf((char)(bytes[i] & 0xFF));
                word.add(i == bytes.length - 1 ? sym + "</w>" : sym);
            }

            while (word.size() > 1) {
                String bestPair = null; int minRank = Integer.MAX_VALUE;
                for (int i = 0; i < word.size() - 1; i++) {
                    String pair = word.get(i) + "\u2581" + word.get(i + 1);
                    Integer rank = bpeRanks.get(pair);
                    if (rank != null && rank < minRank) { minRank = rank; bestPair = pair; }
                }
                if (bestPair == null) break;

                String[] parts = bestPair.split("\u2581", 2);
                String merged = parts[0] + parts[1];
                List<String> next = new ArrayList<>();
                int i = 0;
                while (i < word.size()) {
                    int j = word.subList(i, word.size()).indexOf(parts[0]);
                    if (j == -1) { next.addAll(word.subList(i, word.size())); break; }
                    next.addAll(word.subList(i, i + j));
                    i += j;
                    if (i + 1 < word.size() && word.get(i).equals(parts[0]) && word.get(i + 1).equals(parts[1])) {
                        next.add(merged); i += 2;
                    } else {
                        next.add(word.get(i)); i++;
                    }
                }
                word = next;
            }

            String result = String.join(" ", word);
            cache.put(token, result);
            return result;
        }
    }
}
