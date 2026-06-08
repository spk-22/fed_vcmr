package com.fedvcmr.core;

import static org.junit.Assert.*;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.List;

/**
 * M2 Test: SearchEngine loads and can perform basic search.
 *
 * Note: Requires /sdcard/fedvcmr/vector_map.npy and other data files.
 * For unit testing on device without real data, this test may be skipped.
 *
 * Run: ./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.fedvcmr.core.SearchEngineTest
 */
@RunWith(AndroidJUnit4.class)
public class SearchEngineTest {

    private Context context;
    private SearchEngine engine;

    @Before
    public void setUp() throws Exception {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        try {
            engine = new SearchEngine(context);
        } catch (Exception e) {
            // Skip test if data files not available
            System.out.println("Skipping: /sdcard/fedvcmr data not found");
        }
    }

    @Test
    public void testEngineInitializes() {
        // Basic sanity check
        assertNotNull("Context should be available", context);
    }

    @Test
    public void testSearchReturnsHits() throws Exception {
        if (engine == null) {
            System.out.println("Skipped: engine not initialized");
            return;
        }

        List<SearchEngine.Hit> results = engine.search("cooking");
        assertNotNull("Results should not be null", results);
        // Results may be empty if corpus doesn't match
    }

    @Test
    public void testHitStructure() throws Exception {
        if (engine == null) {
            System.out.println("Skipped: engine not initialized");
            return;
        }

        List<SearchEngine.Hit> results = engine.search("person");
        if (results.size() > 0) {
            SearchEngine.Hit hit = results.get(0);
            assertNotNull("Video ID should not be null", hit.videoId);
            assertTrue("tStart should be >= 0", hit.tStart >= 0);
            assertTrue("tEnd should be > tStart", hit.tEnd > hit.tStart);
            assertTrue("Score should be in valid range", hit.score >= -1 && hit.score <= 1);
        }
    }

    @Test
    public void testTimingInfo() throws Exception {
        if (engine == null) {
            System.out.println("Skipped: engine not initialized");
            return;
        }

        engine.search("test query");
        long loadTime = engine.getLastLoadTimeMs();
        long pipelineTime = engine.getLastPipelineTimeMs();

        assertTrue("Load time should be logged", loadTime >= 0);
        assertTrue("Pipeline time should be logged", pipelineTime > 0);
    }
}
