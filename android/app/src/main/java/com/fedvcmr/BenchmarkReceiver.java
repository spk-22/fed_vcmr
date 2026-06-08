package com.fedvcmr;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

public class BenchmarkReceiver extends BroadcastReceiver {

    private static final String TAG = "BENCH_RESULT";

    @Override
    public void onReceive(Context context, Intent intent) {
        String delegateType = intent.getStringExtra("delegate");
        String modelName = intent.getStringExtra("model");
        int runs = intent.getIntExtra("runs", 10);

        if (delegateType == null) delegateType = "CPU";
        if (modelName == null) modelName = "dgse";

        Log.i(TAG, "Benchmark starting: model=" + modelName + " delegate=" + delegateType + " runs=" + runs);

        try {
            String filename = modelName + ".tflite";
            Interpreter interpreter = ModelLoader.createInterpreter(context, filename, delegateType);

            int inputCount = interpreter.getInputTensorCount();
            int outputCount = interpreter.getOutputTensorCount();
            StringBuilder info = new StringBuilder();
            info.append("inputs=").append(inputCount).append(" outputs=").append(outputCount);
            for (int i = 0; i < inputCount; i++) {
                Tensor t = interpreter.getInputTensor(i);
                info.append(" | in").append(i).append("=").append(shapeStr(t.shape())).append(" (").append(t.numBytes()).append("B)");
            }
            for (int i = 0; i < outputCount; i++) {
                Tensor t = interpreter.getOutputTensor(i);
                info.append(" | out").append(i).append("=").append(shapeStr(t.shape())).append(" (").append(t.numBytes()).append("B)");
            }
            Log.i(TAG, "Model loaded: " + info);

            // Allocate input ByteBuffers
            ByteBuffer[] inputs = new ByteBuffer[inputCount];
            for (int i = 0; i < inputCount; i++) {
                int numBytes = interpreter.getInputTensor(i).numBytes();
                inputs[i] = ByteBuffer.allocateDirect(numBytes);
                inputs[i].order(ByteOrder.nativeOrder());
            }

            // Allocate output ByteBuffers
            Map<Integer, Object> outputs = new HashMap<>();
            for (int i = 0; i < outputCount; i++) {
                int numBytes = interpreter.getOutputTensor(i).numBytes();
                ByteBuffer buf = ByteBuffer.allocateDirect(numBytes);
                buf.order(ByteOrder.nativeOrder());
                outputs.put(i, buf);
            }

            // Warmup (rewind buffers each time)
            for (int w = 0; w < 3; w++) {
                for (ByteBuffer b : inputs) b.rewind();
                for (Object o : outputs.values()) ((ByteBuffer) o).rewind();
                interpreter.runForMultipleInputsOutputs(inputs, outputs);
            }

            // Timed runs
            long totalNs = 0;
            long minNs = Long.MAX_VALUE;
            long maxNs = 0;
            for (int i = 0; i < runs; i++) {
                for (ByteBuffer b : inputs) b.rewind();
                for (Object o : outputs.values()) ((ByteBuffer) o).rewind();

                long start = SystemClock.elapsedRealtimeNanos();
                interpreter.runForMultipleInputsOutputs(inputs, outputs);
                long elapsed = SystemClock.elapsedRealtimeNanos() - start;
                totalNs += elapsed;
                if (elapsed < minNs) minNs = elapsed;
                if (elapsed > maxNs) maxNs = elapsed;
            }

            float avgMs = (totalNs / (float) runs) / 1_000_000f;
            float minMs = minNs / 1_000_000f;
            float maxMs = maxNs / 1_000_000f;

            Log.i(TAG, "model=" + modelName + " | delegate=" + delegateType
                    + " | avg_ms=" + String.format("%.2f", avgMs)
                    + " | min_ms=" + String.format("%.2f", minMs)
                    + " | max_ms=" + String.format("%.2f", maxMs)
                    + " | runs=" + runs);

            interpreter.close();

        } catch (Exception e) {
            Log.e(TAG, "Benchmark failed: " + e.getMessage(), e);
        }
    }

    private static String shapeStr(int[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
