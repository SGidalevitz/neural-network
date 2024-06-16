package Main;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.zip.GZIPInputStream;

public class Datasets {

    public static NeuralNetwork getXOR() {
        int[] networkTopology = {2, 3, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        double[][] input_data = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] output_data = {
                {0},
                {1},
                {1},
                {0}
        };
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    public static NeuralNetwork getOR() {
        int[] networkTopology = {2, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        double[][] input_data = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] output_data = {
                {0},
                {1},
                {1},
                {1}
        };
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    public static NeuralNetwork getAND() {
        int[] networkTopology = {2, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        double[][] input_data = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] output_data = {
                {0},
                {0},
                {0},
                {1}
        };
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    public static NeuralNetwork getNAND() {
        int[] networkTopology = {2, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        double[][] input_data = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] output_data = {
                {1},
                {1},
                {1},
                {0}
        };
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    public static NeuralNetwork getSillyLogicGate() {
        int[] networkTopology = {2, 3, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        double[][] input_data = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] output_data = {
                {0},
                {1},
                {0},
                {1}
        };
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    public static NeuralNetwork getNOT() {
        int[] networkTopology = {1, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        double[][] input_data = {
                {0},
                {1}
        };
        double[][] output_data = {
                {1},
                {0}
        };
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    public static NeuralNetwork getAdder(int bitCount) {
        int inputSize = 2 * bitCount;
        int outputSize = bitCount + 1;
        int[] networkTopology = {inputSize, 2 * inputSize, outputSize};
        NeuralNetwork network = new NeuralNetwork(networkTopology, Activation.Sigmoid, Activation.Sigmoid);
        int numInputs = (int) Math.pow(2, bitCount);
        int numSamples = numInputs * numInputs;

        double[][] input_data = new double[numSamples][inputSize];
        double[][] output_data = new double[numSamples][outputSize];

        int sampleIndex = 0;
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                double[] inputBits = getBits(i, bitCount);
                double[] outputBits = getBits(j, bitCount);

                System.arraycopy(inputBits, 0, input_data[sampleIndex], 0, bitCount);
                System.arraycopy(outputBits, 0, input_data[sampleIndex], bitCount, bitCount);

                int sum = i + j;
                double[] sumBits = getBits(sum, bitCount + 1);
                System.arraycopy(sumBits, 0, output_data[sampleIndex], 0, bitCount + 1);

                sampleIndex++;
            }
        }
        network.setTrainingIO(input_data, output_data);
        return network;
    }
    private static double[] getBits(int number, int bitCount) {
        double[] bits = new double[bitCount];
        for (int i = 0; i < bitCount; i++) {
            bits[bitCount - 1 - i] = (number >> i) & 1;
        }
        return bits;
    }
    public static NeuralNetwork getAddFour() {
        int[] networkTopology = {4, 4, 2, 1};
        double[][][] io = getIOForFourAdder(100);
        NeuralNetwork network = new NeuralNetwork(networkTopology, Activation.Sigmoid, Activation.Sigmoid);
        network.setTrainingIO(io[0], io[1]);
        return network;

    }
    public static void printIO(double[][] inputData, double[][] outputData) {
        System.out.println("Input data = {");
        for (double[] inputDatum : inputData) {
            System.out.print("\t");
            for (int j = 0; j < inputData[0].length; j++) {
                System.out.print(inputDatum[j] + " ");
            }
            System.out.println();
        }
        System.out.println("}\n");
        System.out.println("Output data = {");
        for (double[] outputDatum : outputData) {
            System.out.print("\t");
            for (int j = 0; j < outputData[0].length; j++) {
                System.out.print(outputDatum[j] + " ");
            }
            System.out.println();
        }
        System.out.println("}\n");

    }
    public static double[][][] getIOForFourAdder(int numSamples) {
        double[][] input_data = new double[numSamples][4];
        double[][] output_data = new double[numSamples][1];
        for (int i = 0; i < numSamples; i++) {
            double sum = 0;
            double[] data = new double[4];
            for (int j = 0; j < 4; j++) {
                data[j] = Math.random() / 4;
                sum += data[j];
            }
            input_data[i] = data;
            output_data[i] = new double[]{sum};
        }
        return new double[][][]{input_data, output_data};
    }
    public static NeuralNetwork getMnist() {
        int[] networkTopology = {28 * 28, 16, 16, 10};
        double[][][] IO = getMnistIO();
        NeuralNetwork network = new NeuralNetwork(networkTopology);
        network.setTrainingIO(IO[0], IO[1]);
        return network;
    }
    public static double[][][] getMnistIO() {
        String trainImagesPath = "mnist-data/train-inputs.gz";
        String trainLabelsPath = "mnist-data/train-outputs.gz";
        double[][] inputs = null, outputs = null;
        try {
            InputStream imgIn = new GZIPInputStream(new FileInputStream(trainImagesPath));
            InputStream lblIn = new GZIPInputStream(new FileInputStream(trainLabelsPath));

            byte[] tempBuffer = new byte[16];
            imgIn.read(tempBuffer, 0, 16);
            lblIn.read(tempBuffer, 0, 16);

            byte[] dataBuffer = new byte[1];
            String[] labels = new String[60000];
            float[][][] images = new float[60000][28][28];
            for (int i = 0; i < 60000; i++){
                System.out.printf("Iter: %d/60000\n", i + 1);
                lblIn.read(dataBuffer, 0, 1);
                labels[i] = Integer.toString(dataBuffer[0] & 0xFF);

                for (int j = 0; j < 784; j++){
                    imgIn.read(dataBuffer, 0, 1);
                    float pixelVal = (dataBuffer[0] & 0xFF) / 255.f;
                    images[i][j / 28][j % 28] = pixelVal;
                }
            }
            inputs = new double[60000][28 * 28];
            outputs = new double[labels.length][10];
            for (int i = 0; i < labels.length; i++) {
                int value = Integer.parseInt(labels[i]);
                for (int j = 0; j < 10; j++) {
                    outputs[i][j] = j == value ? 1 : 0;
                }
            }

            for (int i = 0; i < 60000; i++) {
                float[][] arr = images[i];
                double[] input = new double[28 * 28];
                for (int row = 0; row < 28; row++) {
                    for (int col = 0; col < 28; col++) {
                        input[row * 28 + col] = arr[row][col];
                    }
                }
                inputs[i] = input;
            }


        } catch (IOException e) {
            e.printStackTrace();
        }
        return new double[][][]{inputs, outputs};
    }
}
