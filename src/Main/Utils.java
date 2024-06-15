package Main;

public class Utils {
    public static double activationIndividual(double value, Activation activation) {
        return switch (activation) {
            case ReLU -> value > 0 ? value : 0;
            case LeakyReLU -> value > 0 ? value : 0.1 * value;
            case Sigmoid -> 1 / (1 + Math.exp(-value));
            case Tanh -> Math.tanh(value);
        };
    }
    public static double activationDerivativeIndividual(double value, Activation activation) {
        return switch (activation) {
            case ReLU -> value > 0 ? 1 : 0;
            case LeakyReLU -> value > 0 ? 1 : 0.1;
            case Sigmoid -> activationIndividual(value, activation) * (1 - activationIndividual(value, activation)); // It is implied that the activation function will use the Sigmoid equation.
            case Tanh -> 1 - Math.pow(activationIndividual(value, activation), 2); // It is implied that the activation function will use the Tanh equation.
        };
    }
    public static double[] activation(double[] values, Activation activation) {
        Matrix m_values = Matrix.fromOneDimensionalArray(values);
        m_values.apply(value -> activationIndividual(value, activation));
        return Matrix.toOneDimensionalArray(m_values);
    }
    public static double[] activationDerivative(double[] values, Activation activation) {
        Matrix m_values = Matrix.fromOneDimensionalArray(values);
        m_values.apply(value -> activationDerivativeIndividual(value, activation));
        return Matrix.toOneDimensionalArray(m_values);
    }
    public static double softMax(double value, double[] inputs) {
        double sum = 0;
        for (double input : inputs) {
            sum += Math.exp(input);
        }
        return Math.exp(value) / sum;
    }
    public static double dot(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Two vectors of different lengths are being dotted");
        }
        double sum = 0;
        for (int index = 0; index < vec1.length; index++) {
            sum += vec1[index] * vec2[index];
        }
        return sum;
    }

}
