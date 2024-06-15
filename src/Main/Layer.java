package Main;

import java.util.Random;

public class Layer {
    private final int numNeurons;
    public Matrix weights;
    public Matrix biases;
    private final Random random = new Random();
    public Matrix lastWeightChange;
    public Matrix lastBiasChange;

    // Traditional network with unknown weights and biases, must initialize on our own
    public Layer(int numNeurons, int previousLayerNeurons) {
        initializeLastWeightAndBiasChanges(numNeurons, previousLayerNeurons);
        this.numNeurons = numNeurons;
        this.weights = new Matrix(numNeurons, previousLayerNeurons);
        this.biases = new Matrix(numNeurons, 1);
        initializeWeights();
        initializeBiases();
    }

    // Known weights and biases, usually used for testing
    public Layer(int numNeurons, int previousLayerNeurons, Matrix weightMatrix, Matrix biasMatrix) {
        initializeLastWeightAndBiasChanges(numNeurons, previousLayerNeurons);
        if (!(weightMatrix.getNumRows() == numNeurons && weightMatrix.getNumCols() == previousLayerNeurons && biasMatrix.getNumRows() == numNeurons && biasMatrix.getNumCols() == 1)) {
            throw new IllegalArgumentException("Matrix dimensions don't match numNeurons and previousLayerNeurons.");
        }
        this.numNeurons = numNeurons;
        this.weights = weightMatrix;
        this.biases = biasMatrix;
    }
    private void initializeLastWeightAndBiasChanges(int numNeurons, int previousLayerNeurons) {
        this.lastWeightChange = new Matrix(numNeurons, previousLayerNeurons);
        this.lastWeightChange.fill(0.0);

        this.lastBiasChange = new Matrix(numNeurons, 1);
        this.lastBiasChange.fill(0.0);
    }

    private void initializeWeights() {
        // Initialize weights using Xavier initialization
        double sqrt2OverN = Math.sqrt(2.0 / weights.getNumCols()); // Xavier initialization for tanh activation
        for (int row = 0; row < weights.getNumRows(); row++) {
            for (int col = 0; col < weights.getNumCols(); col++) {
                weights.set(row, col, random.nextGaussian() * sqrt2OverN);
            }
        }
    }

    private void initializeBiases() {
        // Initialize biases to small random values
        biases.fillWithRand(-1, 1);
    }

    public int getNumNeurons() {
        return this.numNeurons;
    }

    public Matrix getLastWeightChange() {
        return this.lastWeightChange;
    }

    public Matrix getLastBiasChange() {
        return this.lastBiasChange;
    }
}
