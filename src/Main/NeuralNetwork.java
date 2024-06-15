package Main;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class NeuralNetwork {
    private final int[] layerLengths;
    private final int numLayers;
    private final int inputLength;
    private final int outputLength;
    private final double[][] networkIns;
    public final double[][] networkOuts;
    double[][] trainingInputs;
    double[][] trainingOutputs;
    private static final double LEARNING_RATE = 0.01;
    public static final double EPS = 0.1;
    private static final double MOMENTUM = 0.9;
    private final Activation INNER_ACTIVATION;
    private final Activation OUTER_ACTIVATION;
    private Layer[] layers;
    private int LAST_LAYER;
    private long storedTime;

    public static void main(String[] args) {
        int NUM_TRIALS = 10;
        int NUM_EPOCHS = 10000;
        //NeuralNetwork net = Datasets.getAdder(2);
        //Datasets.printIO(net.trainingInputs, net.trainingOutputs);
        //NeuralNetwork.trainOnce(Datasets.getAdder(4), NUM_EPOCHS, true);
        NeuralNetwork.runBenchmarks(Datasets.getAdder(4), NUM_TRIALS, NUM_EPOCHS);

    }
    public static void trainOnce(NeuralNetwork network, int NUM_EPOCHS, boolean verbose) {
        network.storedTime = System.currentTimeMillis();
        network.train(NUM_EPOCHS, 0, 1E-5, verbose);
        network.printAllTrainingValuesForAdderInDecimal(true);
        network.printTimeDetails(NUM_EPOCHS);
        network.printCostDetails();
    }

    public static void runBenchmarks(NeuralNetwork originalNetwork, int NUM_TRIALS, int NUM_EPOCHS) {
        long initialTime = System.currentTimeMillis();
        double totalCost = 0;
        double trialStartTime;
        for (int i = 0; i < NUM_TRIALS; i++) {
            trialStartTime = System.currentTimeMillis();
            NeuralNetwork network = new NeuralNetwork(originalNetwork.layerLengths);
            network.setTrainingIO(originalNetwork.trainingInputs, originalNetwork.trainingOutputs);
            network.train(NUM_EPOCHS, 0, 1E-5, false);
            totalCost += network.costFunction(network.trainingOutputs, network.feedForward(network.trainingInputs));
            System.out.printf("%d/%d <- %.1f milliseconds\n", i + 1, NUM_TRIALS, System.currentTimeMillis() - trialStartTime);
        }
        long totalTime = System.currentTimeMillis() - initialTime;
        double avgTime = (double) totalTime / NUM_TRIALS;
        double avgCost = totalCost / NUM_TRIALS;
        System.out.printf("Avg time: %f milliseconds\nAvg cost: %f\n", avgTime, avgCost);
    }

    public void printCostDetails() {
        System.out.printf("Cost: %f\n", this.costFunction(trainingOutputs, feedForward(trainingInputs)));
    }

    public void printTimeDetails(int numEpochs) {
        long totalTime = System.currentTimeMillis() - this.storedTime;
        System.out.printf("Total time taken: %,d milliseconds\n", totalTime);
        System.out.printf("Time per epoch: %.3f milliseconds\n", (double)totalTime / numEpochs);
    }

    // For weight initialization


    public NeuralNetwork(int[] layerLengths, Layer[] layers, Activation innerActivation, Activation outerActivation) {
        this.layerLengths = layerLengths;
        this.numLayers = layerLengths.length - 1;
        this.inputLength = layerLengths[0];
        this.outputLength = layerLengths[layerLengths.length - 1];
        this.networkIns = new double[numLayers][];
        this.networkOuts = new double[numLayers + 1][];
        this.INNER_ACTIVATION = innerActivation;
        this.OUTER_ACTIVATION = outerActivation;
        this.LAST_LAYER = numLayers - 1;
        this.storedTime = System.currentTimeMillis();
        initLayers(layers);
    }

    // Known layers, not given activations
    public NeuralNetwork(int[] layerLengths, Layer[] layers) {
        this(layerLengths, layers, Activation.Sigmoid, Activation.Sigmoid);
    }

    // Unknown layers, given activations
    public NeuralNetwork(int[] layerLengths, Activation innerActivation, Activation outerActivation) {
        this(layerLengths, new Layer[layerLengths.length - 1], innerActivation, outerActivation);
    }

    // Unknown layers, not given activations
    public NeuralNetwork(int[] layerLengths) {
        this(layerLengths, new Layer[layerLengths.length - 1], Activation.Sigmoid, Activation.Sigmoid);
    }

    private void initLayers(Layer[] layers) {
        this.layers = layers;
        this.layers[0] = new Layer(layerLengths[1], inputLength);
        for (int i = 0; i < this.layers.length; i++) {
            this.layers[i] = new Layer(layerLengths[i + 1], layerLengths[i]);
        }
    }
    public void setTrainingIO(double[][] inputs, double[][] outputs) {
        this.trainingInputs = inputs;
        this.trainingOutputs = outputs;
    }

    public double[] feedForward(double[] inputs) {
        boolean correctNumberOfInputs = inputs.length == inputLength;
        if (!correctNumberOfInputs) {
            throw new IllegalArgumentException("Invalid number of inputs for forward propagation");
        }
        double[] previousLayerOutputs = null;
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            if (i > 0) inputs = previousLayerOutputs;
            this.networkOuts[i] = inputs;
            Matrix weightMatrix = layer.weights;
            Matrix m_outputs = weightMatrix.dot(Matrix.fromOneDimensionalArray(inputs).transpose());
            m_outputs.add(layer.biases); // wTx + b
            double[] outputs = Matrix.toOneDimensionalArray(m_outputs);
            this.networkIns[i] = outputs;
            previousLayerOutputs = Utils.activation(outputs, getActivation(i == layers.length - 1));
        }
        this.networkOuts[layers.length] = previousLayerOutputs;
        return previousLayerOutputs;
    }
    public double[][] feedForward(double[][] inputs) {
        double[][] outputs = new double[inputs.length][outputLength];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = feedForward(inputs[i]);
        }
        return outputs;
    }

    private Activation getActivation(boolean isOuter) { return isOuter ? OUTER_ACTIVATION : INNER_ACTIVATION; }

    public void trainUsingFiniteDiff() {
        for (int layerIndex = numLayers - 1; layerIndex >= 0; layerIndex--) {
            Layer currentLayer = this.layers[layerIndex];
            Matrix weightsGradient = finiteDiff(layerIndex, Difference.Weights);
            Matrix biasesGradient = finiteDiff(layerIndex, Difference.Biases);
            Matrix dWeights = weightsGradient.dot(LEARNING_RATE);
            Matrix dBiases = biasesGradient.dot(LEARNING_RATE);
            currentLayer.weights.add(dWeights);
            currentLayer.biases.add(dBiases);
        }
    }
    private enum Difference {
        Weights, Biases
    }
    private Matrix finiteDiff(int layerIndex, Difference typeOfDifference) {
        double originalCost = costFunction(trainingOutputs, feedForward(trainingInputs));
        Layer currentLayer = this.layers[layerIndex];
        Matrix currentLayerValues = switch (typeOfDifference) {
            case Weights ->  currentLayer.weights;
            case Biases -> currentLayer.biases;
        };
        Matrix differences = new Matrix(currentLayerValues.getNumRows(), currentLayerValues.getNumCols());
        for (int rowIndex = 0; rowIndex < currentLayerValues.getNumRows(); rowIndex++) {
            for (int colIndex = 0; colIndex < currentLayerValues.getNumCols(); colIndex++) {
                double originalValue = currentLayerValues.get(rowIndex, colIndex);
                currentLayerValues.add(rowIndex, colIndex, EPS);
                double newCost = costFunction(trainingOutputs, feedForward(trainingInputs));
                double difference = (newCost - originalCost) / EPS;
                differences.set(rowIndex, colIndex, -difference);
                currentLayerValues.set(rowIndex, colIndex, originalValue);
            }
        }
        return differences;
    }
    public void backprop() {
        // Initialize gradient accumulators
        Matrix[] weightGradients = new Matrix[numLayers];
        Matrix[] biasGradients = new Matrix[numLayers];

        for (int i = 0; i < numLayers; i++) {
            weightGradients[i] = new Matrix(layers[i].weights.getNumRows(), layers[i].weights.getNumCols());
            biasGradients[i] = new Matrix(layers[i].biases.getNumRows(), layers[i].biases.getNumCols());
        }

        int numSamples = trainingInputs.length;

        // Accumulate gradients for each training sample
        for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            double[] inputs = trainingInputs[sampleIndex];
            double[] expectedOutputs = trainingOutputs[sampleIndex];
            double[] actualOutputs = feedForward(inputs);
            checkArgumentValidityForBackprop(inputs, expectedOutputs);

            // Compute the error at the output layer
            double[] outputLayerError = new double[outputLength];
            for (int i = 0; i < outputLength; i++) {
                outputLayerError[i] = actualOutputs[i] - expectedOutputs[i];
            }

            // Backpropagate the error
            double[][] errors = new double[numLayers][];
            errors[numLayers - 1] = outputLayerError;

            for (int layerIndex = numLayers - 1; layerIndex > 0; layerIndex--) {
                errors[layerIndex - 1] = new double[layerLengths[layerIndex]];
                for (int j = 0; j < layerLengths[layerIndex]; j++) {
                    double error = 0.0;
                    for (int k = 0; k < layerLengths[layerIndex + 1]; k++) {
                        error += errors[layerIndex][k] * layers[layerIndex].weights.get(k, j);
                    }
                    errors[layerIndex - 1][j] = error * Utils.activationDerivativeIndividual(networkIns[layerIndex - 1][j], getActivation(false));
                }
            }

            // Compute gradients for weights and biases
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
                for (int j = 0; j < layers[layerIndex].weights.getNumRows(); j++) {
                    for (int k = 0; k < layers[layerIndex].weights.getNumCols(); k++) {
                        double input = layerIndex == 0 ? inputs[k] : networkOuts[layerIndex][k];
                        weightGradients[layerIndex].add(j, k, errors[layerIndex][j] * input);
                    }
                }
                for (int j = 0; j < layers[layerIndex].biases.getNumRows(); j++) {
                    biasGradients[layerIndex].add(j, 0, errors[layerIndex][j]);
                }
            }
        }

        // Average gradients and update weights and biases
        for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
            weightGradients[layerIndex].dot(1.0 / numSamples);
            biasGradients[layerIndex].dot(1.0 / numSamples);
            layers[layerIndex].weights.subtract(weightGradients[layerIndex].dot(LEARNING_RATE));
            layers[layerIndex].biases.subtract(biasGradients[layerIndex].dot(LEARNING_RATE));
        }
    }
    public void checkArgumentValidityForBackprop(double[] inputs, double[] expectedOutputs) {
        boolean correctNumberOfOutputs = expectedOutputs.length == this.outputLength;
        boolean correctNumberOfInputs  =          inputs.length == this.inputLength;
        if (!correctNumberOfOutputs) {
            throw new IllegalArgumentException("Number of outputs is wrong for backpropagation method");
        }
        if (!correctNumberOfInputs) {
            throw new IllegalArgumentException("Number of inputs is wrong for backpropagation method");
        }
    }

    // Mean Squared Error Cost function
    public double costFunction(double[] expectedOutputs, double[] actualOutputs) {
        boolean correctNumberOfOutputs = expectedOutputs.length == outputLength;
        if (!correctNumberOfOutputs) {
            throw new IllegalArgumentException("Invalid number of outputs for cost function");
        }
        double sum = 0;
        for (int i = 0; i < outputLength; i++) {
            sum += Math.pow((expectedOutputs[i] - actualOutputs[i]), 2);
        }
        sum /= (2 * outputLength);
        return sum;
    }
    public double costFunction(double[][] expectedOutputs, double[][] actualOutputs) {
        double totalTrainingExamples = expectedOutputs.length;
        double totalCost = IntStream.range(0, expectedOutputs.length).mapToDouble(i -> costFunction(expectedOutputs[i], actualOutputs[i])).sum();
        return totalCost / totalTrainingExamples;
    }
    public double[] derivativeCostFunction(double[] expectedOutputs, double[] actualOutputs) {
        Matrix m_actual = Matrix.fromOneDimensionalArray(actualOutputs);
        Matrix m_expected = Matrix.fromOneDimensionalArray(expectedOutputs);
        Matrix result = Matrix.difference(m_expected, m_actual);
        return Matrix.toOneDimensionalArray(result);
    }
    public void train(int epochs, int batchSize, double errorThreshold, boolean verbose) {
        int numberOfInputs = trainingInputs.length;
        //int trainingSetIndex = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            this.backprop();
            if (epoch % 100 == 0 && costFunction(this.trainingOutputs, this.feedForward(this.trainingInputs)) < errorThreshold) {
                if (verbose) {
                    System.out.println("[TRAIN]: Training ended early because error threshold was reached.");
                }
                break;
            }
            if (verbose) {
                System.out.printf("[TRAIN]: Cost on epoch %d/%d: %f\n", epoch + 1, epochs, costFunction(feedForward(trainingInputs), trainingOutputs));
            }
        }

    }
    public void printAllTrainingValuesForAdderInDecimal(boolean onlyWrong) {
        int wrong = 0;
        int numBits = (int)(Math.log(trainingInputs.length)/Math.log(2)) / 2;
        int overflow = (int)Math.pow(2, numBits);
        for (double[] inputs : trainingInputs) {
            double[] input1A = new double[inputs.length / 2];
            double[] input2A = new double[inputs.length / 2];
            for (int j = 0; j < inputs.length; j++) {
                if (j >= inputs.length / 2) {
                    input2A[j - inputs.length / 2] = inputs[j];
                } else {
                    input1A[j] = inputs[j];
                }
            }
            int input1 = binaryToDecimal(input1A);
            int input2 = binaryToDecimal(input2A);
            double[] networkOutput = this.feedForward(inputs);
            int output = binaryToDecimal(networkOutput);
            String outputS = "" + output;

            if (output != input1 + input2) {
                wrong++;
            }
            if (onlyWrong) {
                if (output != input1 + input2) {
                    System.out.println(input1 + " + " + input2 + " ?= " + outputS);
                }
            } else {
                System.out.println(input1 + " + " + input2 + " ?= " + outputS);
            }

        }
        System.out.printf("Accuracy: %d/%d\n", trainingInputs.length - wrong, trainingInputs.length);
    }

    private static int binaryToDecimal(double[] binary) {
        int decimal = 0;
        for (int i = 0; i < binary.length; i++) {
            binary[i] = Math.round(binary[i]);
            decimal += (int) (binary[i] * Math.pow(2, binary.length - 1 - i));
        }
        return decimal;
    }
    public void printAllTrainingValues() {
        int count = 0;
        for (int i = 0; i < trainingInputs.length; i++) {
            double[] inputs = trainingInputs[i];
            printTrainingWithoutRounding(inputs);
            if (Arrays.equals(trainingOutputs[i], Arrays.stream(this.feedForward(inputs)).map(this::halfAct).toArray())) count++;
        }
        System.out.println("Accuracy: " + 100 * (double)count / trainingInputs.length + "%\n");
    }
    public void printTrainingWithRounding(double[] inputs) {
        System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(Arrays.stream(this.feedForward(inputs)).map(this::halfAct).toArray()));
    }
    public void printTrainingWithoutRounding(double[] inputs) {
        System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(this.feedForward(inputs)));
    }
    private int halfAct(double value) {
        return value > 0.5 ? 1 : 0;
    }
    public static void debugPrintln(String str) {
        System.out.println("[DEBUG]: " + str);
    }
}
