package com.fedvcmr;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Loads TFLite models from assets/ with configurable delegate (CPU, GPU, NNAPI).
 */
public class ModelLoader {

    private static final String TAG = "ModelLoader";

    public static MappedByteBuffer loadModelFile(Context context, String filename) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd("models/" + filename);
        FileInputStream fis = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = fis.getChannel();
        long startOffset = fd.getStartOffset();
        long declaredLength = fd.getDeclaredLength();
        return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static Interpreter createInterpreter(Context context, String filename, String delegate) throws IOException {
        MappedByteBuffer model = loadModelFile(context, filename);
        Interpreter.Options options = new Interpreter.Options();

        switch (delegate.toUpperCase()) {
            case "GPU":
                Log.i(TAG, "Using GPU delegate (Adreno 730)");
                options.addDelegate(new GpuDelegate());
                break;
            case "NNAPI":
                Log.i(TAG, "Using NNAPI delegate (Hexagon 780)");
                options.addDelegate(new NnApiDelegate());
                break;
            default:
                Log.i(TAG, "Using CPU delegate (XNNPACK)");
                break;
        }

        return new Interpreter(model, options);
    }
}
