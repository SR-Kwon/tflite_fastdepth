package com.example.fd_tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // view 선언
        ImageView target = findViewById(R.id.activity_main__target);
        ImageView result = findViewById(R.id.activity_main__depthimg);

        // 이미지 읽어오기
        Bitmap input_img = BitmapFactory.decodeResource(getResources(), R.drawable.test_converted);
        target.setImageBitmap(input_img);

        float[] input_array;
        try {
            input_array = loadImageAsFloatArray(input_img);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        float[][][][] model_input_array = arrayToModelInput(input_array);

        // TensorFlow Lite 모델 초기화
        Interpreter tflite = getTfliteInterpreter("fastestdepth_float32.tflite");

        // 모델 돌리기
        float[][][][] output_array = new float[1][224][224][1]; // GrayScale

        assert tflite != null;
        tflite.run(model_input_array, output_array);

        // 결과 비트맵
        Bitmap output_img = convertArrayToBitmap(output_array, 224, 224);

        result.setImageBitmap(output_img);

    }

    private float[][][][] arrayToModelInput(float[] input_array){
        float[][][][] final_input = new float[1][224][224][3];

        for (int i = 0; i < 224; i++) {
            for (int j = 0; j < 224; j++) {
                final_input[0][i][j][0] = input_array[(i * 224 + j) * 3];     // R
                final_input[0][i][j][1] = input_array[(i * 224 + j) * 3 + 1]; // G
                final_input[0][i][j][2] = input_array[(i * 224 + j) * 3 + 2]; // B
            }
        }
        return final_input;
    }

    private float[] loadImageAsFloatArray(Bitmap bitmap) throws IOException {
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        return bitmapToFloatArray(bitmap);
    }

    private float[] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[] floatArray = new float[width * height * 3];

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            floatArray[i * 3] = r;
            floatArray[i * 3 + 1] = g;
            floatArray[i * 3 + 2] = b;
        }
        return floatArray;
    }

    private Bitmap convertArrayToBitmap(float[][][][] imageArray, int imageWidth, int imageHeight) {
        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap styledImage = Bitmap.createBitmap(imageWidth, imageHeight, conf);

        float minVal = Float.MAX_VALUE;
        float maxVal = Float.MIN_VALUE;

        for (int x = 0; x < imageArray[0].length; x++) {
            for (int y = 0; y < imageArray[0][0].length; y++) {
                float value = imageArray[0][x][y][0];
                minVal = Math.min(minVal, value);
                maxVal = Math.max(maxVal, value);
            }
        }

        for (int x = 0; x < imageArray[0].length; x++) {
            for (int y = 0; y < imageArray[0][0].length; y++) {
                float normalizedValue = (imageArray[0][x][y][0] - minVal) / (maxVal - minVal);
                int grayValue = (int) (normalizedValue * 255);
                int color = Color.rgb(grayValue, grayValue, grayValue);
                styledImage.setPixel(y, x, color);
            }
        }
        return styledImage;
    }

    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
