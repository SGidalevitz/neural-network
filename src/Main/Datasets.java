package Main;

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
}
