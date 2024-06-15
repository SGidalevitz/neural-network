package Tests;

import Main.Activation;
import Main.Layer;
import Main.Matrix;
import Main.NeuralNetwork;
import org.junit.Assert;
import org.junit.Test;

import java.util.Optional;

public class NeuralNetworkTest {
    public static final double TOLERANCE = 0.0001;
    @Test
    public void testFeedForward() {
        // Set up the weights and biases for the first layer
        Matrix firstLayerWeights = new Matrix(new double[][]{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}});
        Matrix firstLayerBiases = new Matrix(new double[][]{{0.1}, {0.2}, {0.3}});

        // Set up the weights and biases for the second layer
        Matrix secondLayerWeights = new Matrix(new double[][]{{0.7, 0.8, 0.9}});
        Matrix secondLayerBiases = new Matrix(new double[][]{{0.4}});

        // Create the layers
        Layer firstLayer = new Layer(3, 2, firstLayerWeights, firstLayerBiases);
        Layer secondLayer = new Layer(1, 3, secondLayerWeights, secondLayerBiases);
        Layer[] layers = {firstLayer, secondLayer};

        // Create the neural network
        int[] networkTopology = {2, 3, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology, layers, Activation.ReLU, Activation.ReLU);

        // Test the feedForward method
        double[] inputs = {1.0, 1.0};
        double[] actualOutputs = network.feedForward(inputs);

        // Compute the expected outputs
        double[] expectedOutputs = {2.66}; // Replace this with the actual expected outputs

        // Check if the actual outputs match the expected outputs
        Assert.assertArrayEquals(expectedOutputs, actualOutputs, TOLERANCE);
    }
    @Test
    public void testFeedForwardBigNetwork() {
        // Set up the weights and biases for the first layer
        Matrix firstLayerWeights = new Matrix(new double[][]{
                {0.1, 0.2},
                {0.3, 0.4},
                {0.5, 0.6},
                {0.7, 0.8}
        });
        Matrix firstLayerBiases = new Matrix(new double[][]{
                {0.1}, {0.2}, {0.3}, {0.4}
        });

// Set up the weights and biases for the second layer
        Matrix secondLayerWeights = new Matrix(new double[][]{
                {0.9, 0.8, 0.7, 0.6},
                {0.5, 0.4, 0.3, 0.2},
                {0.1, 0.2, 0.3, 0.4}
        });
        Matrix secondLayerBiases = new Matrix(new double[][]{
                {0.5}, {0.6}, {0.7}
        });

// Set up the weights and biases for the third layer
        Matrix thirdLayerWeights = new Matrix(new double[][]{
                {0.9, 0.8, 0.7},
                {0.6, 0.5, 0.4}
        });
        Matrix thirdLayerBiases = new Matrix(new double[][]{
                {0.8}, {0.9}
        });

// Set up the weights and biases for the fourth (output) layer
        Matrix fourthLayerWeights = new Matrix(new double[][]{
                {0.3, 0.2}
        });
        Matrix fourthLayerBiases = new Matrix(new double[][]{
                {0.1}
        });

// Create the layers
        Layer firstLayer = new Layer(4, 2, firstLayerWeights, firstLayerBiases);
        Layer secondLayer = new Layer(3, 4, secondLayerWeights, secondLayerBiases);
        Layer thirdLayer = new Layer(2, 3, thirdLayerWeights, thirdLayerBiases);
        Layer fourthLayer = new Layer(1, 2, fourthLayerWeights, fourthLayerBiases);
        Layer[] layers = {firstLayer, secondLayer, thirdLayer, fourthLayer};

// Create the neural network
        int[] networkTopology = {2, 4, 3, 2, 1};
        NeuralNetwork network = new NeuralNetwork(networkTopology, layers, Activation.ReLU, Activation.ReLU);

// Test the feedForward method
        double[] inputs = {1.0, 1.0};
        double[] actualOutputs = network.feedForward(inputs);
        double[] expectedOutputs = {3.2384};
        Assert.assertArrayEquals(expectedOutputs, actualOutputs, TOLERANCE);
    }
    @Test
    public void testFeedForwardMultipleOutputs() {
        // Set up the weights and biases for the first layer
        Matrix firstLayerWeights = new Matrix(new double[][]{
                {0.1, -0.2},
                {-0.3, 0.4},
                {0.5, -0.6},
                {-0.7, 0.8}
        });
        Matrix firstLayerBiases = new Matrix(new double[][]{
                {0.1}, {-0.2}, {0.3}, {-0.4}
        });

        Matrix secondLayerWeights = new Matrix(new double[][]{
                {0.9, -0.8, 0.7, -0.6},
                {-0.5, 0.4, -0.3, 0.2},
                {0.1, -0.2, 0.3, -0.4}
        });
        Matrix secondLayerBiases = new Matrix(new double[][]{
                {0.5}, {-0.6}, {0.7}
        });

        Matrix thirdLayerWeights = new Matrix(new double[][]{
                {0.9, -0.8, 0.7},
                {-0.6, 0.5, -0.4}
        });
        Matrix thirdLayerBiases = new Matrix(new double[][]{
                {0.8}, {-0.9}
        });

// Set up the weights and biases for the fourth (output) layer
        Matrix fourthLayerWeights = new Matrix(new double[][]{
                {0.3, -0.2},
                {-0.4, 0.3}
        });
        Matrix fourthLayerBiases = new Matrix(new double[][]{
                {0.1}, {-0.2}
        });

        Layer firstLayer = new Layer(4, 2, firstLayerWeights, firstLayerBiases);
        Layer secondLayer = new Layer(3, 4, secondLayerWeights, secondLayerBiases);
        Layer thirdLayer = new Layer(2, 3, thirdLayerWeights, thirdLayerBiases);
        Layer fourthLayer = new Layer(2, 2, fourthLayerWeights, fourthLayerBiases);
        Layer[] layers = {firstLayer, secondLayer, thirdLayer, fourthLayer};

        int[] networkTopology = {2, 4, 3, 2, 2};
        NeuralNetwork network = new NeuralNetwork(networkTopology, layers, Activation.ReLU, Activation.ReLU);

        double[] inputs = {1.0, 1.0};
        double[] actualOutputs = network.feedForward(inputs);
        double[] expectedOutputs = {0.6724, 0.0};

        Assert.assertArrayEquals(expectedOutputs, actualOutputs, TOLERANCE);
    }

    @Test
    public void testCostFunction1() {
        NeuralNetwork network = new NeuralNetwork(new int[]{5, 3, 5});
        double[] expectedOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] actualOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double dividend = 2 * expectedOutputs.length;
        double sum = 0;
        for (int i = 0; i < expectedOutputs.length; i++) {
            sum += Math.pow(expectedOutputs[i] - actualOutputs[i], 2);
        }
        double expectedCost = sum / dividend;
        Assert.assertEquals(expectedCost, network.costFunction(expectedOutputs, actualOutputs), TOLERANCE);
    }
    @Test
    public void testCostFunction2() {
        NeuralNetwork network = new NeuralNetwork(new int[]{5, 3, 5});
        double[] expectedOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] actualOutputs = {0.2, 0.3, 0.4, 0.5, 0.6};
        double dividend = 2 * expectedOutputs.length;
        double sum = 0;
        for (int i = 0; i < expectedOutputs.length; i++) {
            sum += Math.pow(expectedOutputs[i] - actualOutputs[i], 2);
        }
        double expectedCost = sum / dividend;
        Assert.assertEquals(expectedCost, network.costFunction(expectedOutputs, actualOutputs), TOLERANCE);
    }
    @Test
    public void testCostFunction3() {
        NeuralNetwork network = new NeuralNetwork(new int[]{5, 3, 5});
        double[] expectedOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] actualOutputs = {0.5, 0.4, 0.3, 0.2, 0.1};
        double dividend = 2 * expectedOutputs.length;
        double sum = 0;
        for (int i = 0; i < expectedOutputs.length; i++) {
            sum += Math.pow(expectedOutputs[i] - actualOutputs[i], 2);
        }
        double expectedCost = sum / dividend;
        Assert.assertEquals(expectedCost, network.costFunction(expectedOutputs, actualOutputs), TOLERANCE);
    }

    @Test
    public void testDerivativeCostFunction1() {
        NeuralNetwork network = new NeuralNetwork(new int[]{5, 3, 5});
        double[] expectedOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] actualOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] expectedDerivativeCost = {0.0, 0.0, 0.0, 0.0, 0.0};
        Assert.assertArrayEquals(expectedDerivativeCost, network.derivativeCostFunction(expectedOutputs, actualOutputs), TOLERANCE);
    }
    @Test
    public void testDerivativeCostFunction2() {
        NeuralNetwork network = new NeuralNetwork(new int[]{5, 3, 5});
        double[] expectedOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] actualOutputs = {0.2, 0.3, 0.4, 0.5, 0.6};
        double[] expectedDerivativeCost = {-0.1, -0.1, -0.1, -0.1, -0.1};
        Assert.assertArrayEquals(expectedDerivativeCost, network.derivativeCostFunction(expectedOutputs, actualOutputs), TOLERANCE);
    }
    @Test
    public void testDerivativeCostFunction3() {
        NeuralNetwork network = new NeuralNetwork(new int[]{5, 3, 5});
        double[] expectedOutputs = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] actualOutputs = {0.5, 0.4, 0.3, 0.2, 0.1};
        double[] expectedDerivativeCost = {-0.4, -0.2, 0.0, 0.2, 0.4};
        Assert.assertArrayEquals(expectedDerivativeCost, network.derivativeCostFunction(expectedOutputs, actualOutputs), TOLERANCE);
    }

}