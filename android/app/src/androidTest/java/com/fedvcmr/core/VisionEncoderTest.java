package com.fedvcmr.core;

import static org.junit.Assert.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * M1 Test: VisionEncoder loads and produces normalized embeddings from images.
 *
 * Run: ./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.fedvcmr.core.VisionEncoderTest
 */
@RunWith(AndroidJUnit4.class)
public class VisionEncoderTest {

    private Context context;
    private VisionEncoder encoder;

    @Before
    public void setUp() throws Exception {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        encoder = new VisionEncoder(context);
    }

    @Test
    public void testEncoderLoads() {
        assertNotNull("VisionEncoder should be initialized", encoder);
    }

    @Test
    public void testEncodeBitmap() throws Exception {
        // Create a simple test bitmap
        Bitmap bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
        bitmap.eraseColor(Color.GRAY);

        float[] embedding = encoder.encode(bitmap);

        assertNotNull("Embedding should not be null", embedding);
        assertEquals("Embedding should be 512-dimensional", 512, embedding.length);

        bitmap.recycle();
    }

    @Test
    public void testEmbeddingIsNormalized() throws Exception {
        Bitmap bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
        bitmap.eraseColor(Color.RED);

        float[] embedding = encoder.encode(bitmap);

        float norm = 0;
        for (float v : embedding) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);

        assertTrue("Embedding should be normalized (L2 norm ≈ 1)", Math.abs(norm - 1.0f) < 0.01f);

        bitmap.recycle();
    }

    @Test
    public void testEncodeBatch() throws Exception {
        Bitmap[] bitmaps = new Bitmap[3];
        bitmaps[0] = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
        bitmaps[0].eraseColor(Color.RED);
        bitmaps[1] = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
        bitmaps[1].eraseColor(Color.GREEN);
        bitmaps[2] = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
        bitmaps[2].eraseColor(Color.BLUE);

        float[] embeddings = encoder.encodeBatch(bitmaps);

        assertNotNull("Embeddings should not be null", embeddings);
        assertEquals("Should have 3*512 floats", 3 * 512, embeddings.length);

        // Check each embedding is normalized
        for (int b = 0; b < 3; b++) {
            float norm = 0;
            for (int d = 0; d < 512; d++) {
                float v = embeddings[b * 512 + d];
                norm += v * v;
            }
            norm = (float) Math.sqrt(norm);
            assertTrue("Embedding " + b + " should be normalized", Math.abs(norm - 1.0f) < 0.01f);
        }

        for (Bitmap b : bitmaps) {
            b.recycle();
        }
    }

    @Test
    public void testResize() throws Exception {
        // Test with non-224x224 image (should be resized internally)
        Bitmap bitmap = Bitmap.createBitmap(448, 448, Bitmap.Config.ARGB_8888);
        bitmap.eraseColor(Color.BLUE);

        float[] embedding = encoder.encode(bitmap);

        assertNotNull("Embedding should not be null", embedding);
        assertEquals("Embedding should be 512-dimensional even after resize", 512, embedding.length);

        bitmap.recycle();
    }
}
