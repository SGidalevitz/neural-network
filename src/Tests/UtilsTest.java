package Tests;


import Main.Activation;
import org.junit.Assert;
import org.junit.Test;
import Main.Utils;

public class UtilsTest {
    public static final double TOLERANCE = 0.0001;
    @Test
    public void reLUWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, Double.MIN_VALUE};
        double[] expectedOutput = {5.2, 0, 0, 0, 7.78, 0, 0, 1100, Double.MAX_VALUE, 0};
        Assert.assertArrayEquals(expectedOutput, Utils.activation(input, Activation.ReLU), TOLERANCE);
    }
    @Test
    public void leakyReLUWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, Double.MIN_VALUE};
        double[] expectedOutput = {5.2, -0.586, 0, -0.312, 7.78, -0.3141592653589793, -0.87373, 1100, Double.MAX_VALUE, 0};
        Assert.assertArrayEquals(expectedOutput, Utils.activation(input, Activation.LeakyReLU), TOLERANCE);
    }

    @Test
    public void sigmoidWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, Double.MIN_VALUE};
        double[] expectedOutput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            expectedOutput[i] = 1 / (1 + Math.exp(-input[i]));
        }
        Assert.assertArrayEquals(expectedOutput, Utils.activation(input, Activation.Sigmoid), TOLERANCE);
    }
    @Test
    public void tanhWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, Double.MIN_VALUE};
        double[] expectedOutput = {Math.tanh(5.2), Math.tanh(-5.86), Math.tanh(0), Math.tanh(-3.12), Math.tanh(7.78), Math.tanh(-1 * Math.PI), Math.tanh(-8.7373), Math.tanh(1100), 1.0, 0.0};
        Assert.assertArrayEquals(expectedOutput, Utils.activation(input, Activation.Tanh), TOLERANCE);
    }
    @Test
    public void reLUDerivativeWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, -Double.MAX_VALUE};
        double[] expectedOutput = {1, 0, 0, 0, 1, 0, 0, 1, 1, 0};
        Assert.assertArrayEquals(expectedOutput, Utils.activationDerivative(input, Activation.ReLU), TOLERANCE);
    }

    @Test
    public void leakyReLUDerivativeWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, -Double.MAX_VALUE};
        double[] expectedOutput = {1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 1, 1, 0.1};
        Assert.assertArrayEquals(expectedOutput, Utils.activationDerivative(input, Activation.LeakyReLU), TOLERANCE);
    }

    @Test
    public void sigmoidDerivativeWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, Double.MIN_VALUE};
        double[] expectedOutput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double sigmoid = 1 / (1 + Math.exp(-input[i]));
            expectedOutput[i] = sigmoid * (1 - sigmoid);
        }
        Assert.assertArrayEquals(expectedOutput, Utils.activationDerivative(input, Activation.Sigmoid), TOLERANCE);
    }

    @Test
    public void tanhDerivativeWithArray() {
        double[] input = {5.2, -5.86, 0, -3.12, 7.78, -1 * Math.PI, -8.7373, 1100, Double.MAX_VALUE, Double.MIN_VALUE};
        double[] expectedOutput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double tanh = Math.tanh(input[i]);
            expectedOutput[i] = 1 - tanh * tanh;
        }
        Assert.assertArrayEquals(expectedOutput, Utils.activationDerivative(input, Activation.Tanh), TOLERANCE);
    }
    @Test
    public void testDot() {
        double[] array1 = {1.0, 2.0, 3.0};
        double[] array2 = {4.0, 5.0, 6.0};
        double expectedDotProduct = 32.0; // 1*4 + 2*5 + 3*6
        Assert.assertEquals(expectedDotProduct, Utils.dot(array1, array2), 0.0);
    }


}