package com.fedvcmr.core;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.fedvcmr.ModelLoader;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * MobileCLIP-S1 Vision Encoder using PyTorch Mobile (.ptl).
 */
public class VisionEncoder {

    private static final String TAG = "VisionEncoder";

    private static final int IMAGE_SIZE = 256;
    private static final int EMBED_DIM  = 512;

    // MobileCLIP-S1 pretrained on DataCompDR uses [0, 1] scaling only
    private static final float MEAN_R = 0.0f;
    private static final float MEAN_G = 0.0f;
    private static final float MEAN_B = 0.0f;
    private static final float STD_R  = 1.0f;
    private static final float STD_G  = 1.0f;
    private static final float STD_B  = 1.0f;

    private Module module;

    public VisionEncoder(Context context) throws Exception {
        // Load aligned PyTorch Mobile backbone from assets
        this.module = LiteModuleLoader.loadModuleFromAsset(
                context.getAssets(), "models/mobileclip_vision_encoder.ptl");
        Log.i(TAG, "Loaded PyTorch Vision Encoder (MobileCLIP-S1 Backbone)");
    }

    public float[] encodeBatch(Bitmap[] bitmaps) {
        int batch = bitmaps.length;
        float[] result = new float[batch * EMBED_DIM];

        for (int b = 0; b < batch; b++) {
            Bitmap bm = bitmaps[b];
            Bitmap resized = (bm.getWidth() == IMAGE_SIZE && bm.getHeight() == IMAGE_SIZE)
                    ? bm : Bitmap.createScaledBitmap(bm, IMAGE_SIZE, IMAGE_SIZE, true);

            // Preprocess to CHW float array [0, 1]
            float[] floatArray = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
            int[] pixels = new int[IMAGE_SIZE * IMAGE_SIZE];
            resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
            if (resized != bm) resized.recycle();

            for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
                int px = pixels[i];
                floatArray[i] = ((px >> 16) & 0xFF) / 255.0f;
                floatArray[IMAGE_SIZE * IMAGE_SIZE + i] = ((px >> 8) & 0xFF) / 255.0f;
                floatArray[2 * IMAGE_SIZE * IMAGE_SIZE + i] = (px & 0xFF) / 255.0f;
            }

            // Run inference
            Tensor input = Tensor.fromBlob(floatArray, new long[]{1, 3, IMAGE_SIZE, IMAGE_SIZE});
            float[] backbone = module.forward(IValue.from(input)).toTensor().getDataAsFloatArray();

            // Normalize to unit vector
            float norm = 0;
            for (int d = 0; d < EMBED_DIM; d++) norm += backbone[d] * backbone[d];
            norm = (float) Math.sqrt(norm);
            for (int d = 0; d < EMBED_DIM; d++) {
                result[b * EMBED_DIM + d] = backbone[d] / (norm + 1e-10f);
            }
        }

        return result;
    }

    public void close() {
        // PyTorch LiteModule doesn't have an explicit close
    }
}
