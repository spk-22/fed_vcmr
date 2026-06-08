package com.fedvcmr.core;

import static org.junit.Assert.*;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * M1 Test: TextEncoder loads and produces normalized embeddings.
 *
 * Run: ./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.fedvcmr.core.TextEncoderTest
 */
@RunWith(AndroidJUnit4.class)
public class TextEncoderTest {

    private Context context;
    private TextEncoder encoder;

    @Before
    public void setUp() throws Exception {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        encoder = new TextEncoder(context);
    }

    @Test
    public void testEncoderLoads() {
        assertNotNull("TextEncoder should be initialized", encoder);
    }

    @Test
    public void testEncodeProducesVector() throws Exception {
        float[] embedding = encoder.encode("a person cooking pasta");

        assertNotNull("Embedding should not be null", embedding);
        assertEquals("Embedding should be 512-dimensional", 512, embedding.length);
    }

    @Test
    public void testEmbeddingIsNormalized() throws Exception {
        float[] embedding = encoder.encode("cooking");

        float norm = 0;
        for (float v : embedding) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);

        assertTrue("Embedding should be normalized (L2 norm ≈ 1)", Math.abs(norm - 1.0f) < 0.01f);
    }

    @Test
    public void testMultipleEncodings() throws Exception {
        String[] queries = {
            "a person driving a car",
            "cooking pasta",
            "playing soccer",
        };

        for (String query : queries) {
            float[] emb = encoder.encode(query);
            assertNotNull("Embedding should not be null for: " + query, emb);
            assertEquals("Each embedding should be 512-dimensional", 512, emb.length);
        }
    }
}
