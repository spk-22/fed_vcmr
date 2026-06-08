package com.fedvcmr;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Minimal .npy file reader for float16 and float32 arrays.
 * Supports 1D and 2D arrays only.
 */
public class NpyReader {

    /**
     * Read a .npy file and return float32 data.
     * float16 files are automatically converted to float32.
     * Returns flat float array; caller must reshape.
     */
    public static float[] readNpy(String path) throws IOException {
        FileInputStream fis = new FileInputStream(path);
        byte[] header = new byte[10];
        fis.read(header);

        // Verify magic: \x93NUMPY
        if (header[0] != (byte) 0x93 || header[1] != 'N') {
            fis.close();
            throw new IOException("Not a .npy file: " + path);
        }

        // Header length is at bytes 8-9 (little-endian)
        int headerLen = (header[8] & 0xFF) | ((header[9] & 0xFF) << 8);
        byte[] headerData = new byte[headerLen];
        fis.read(headerData);
        String headerStr = new String(headerData).trim();

        // Parse dtype
        boolean isFloat16 = headerStr.contains("'<f2'") || headerStr.contains("'f2'");
        boolean isFloat32 = headerStr.contains("'<f4'") || headerStr.contains("'f4'");

        if (!isFloat16 && !isFloat32) {
            fis.close();
            throw new IOException("Unsupported dtype in: " + headerStr);
        }

        // Read remaining bytes as data
        byte[] data = fis.readAllBytes();
        fis.close();

        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        if (isFloat32) {
            int count = data.length / 4;
            float[] result = new float[count];
            buf.asFloatBuffer().get(result);
            return result;
        } else {
            // float16 -> float32 conversion
            int count = data.length / 2;
            float[] result = new float[count];
            for (int i = 0; i < count; i++) {
                short bits = buf.getShort();
                result[i] = halfToFloat(bits);
            }
            return result;
        }
    }

    /** Convert IEEE 754 half-precision to single-precision. */
    private static float halfToFloat(short bits) {
        int s = (bits >> 15) & 0x1;
        int e = (bits >> 10) & 0x1F;
        int m = bits & 0x3FF;

        if (e == 0) {
            if (m == 0) return s == 0 ? 0.0f : -0.0f;
            // Denormalized
            float val = (float) (m / 1024.0 * Math.pow(2, -14));
            return s == 0 ? val : -val;
        } else if (e == 31) {
            return m == 0 ? (s == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) : Float.NaN;
        }

        float val = (float) ((1.0 + m / 1024.0) * Math.pow(2, e - 15));
        return s == 0 ? val : -val;
    }
}
