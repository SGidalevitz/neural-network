package Main;

import java.util.Random;
import java.util.function.Function;

public class Matrix {
    private final double[][] arr;
    private final int numRows;
    private final int numCols;
    public Matrix(int rows, int cols) {
        this.arr = new double[rows][cols];
        this.numRows = rows;
        this.numCols = cols;
    }
    public Matrix(double[][] arr) {
        this.arr = arr;
        this.numRows = arr.length;
        this.numCols = arr[0].length;
    }
    public Matrix(Matrix other) {
        this(other.toArray());
    }

    public double get(int rowIndex, int colIndex) {
        return this.arr[rowIndex][colIndex];
    }
    public void set(int rowIndex, int colIndex, double value) {
        this.arr[rowIndex][colIndex] = value;
    }
    public double[][] toArray() {
        return arr;
    }

    public void fill(double value) {
        for (int i = 0; i < getNumRows(); i++) {
            for (int j = 0; j < getNumCols(); j++) {
                set(i, j, value);
            }
        }
    }
    public void add(Matrix other) {
        boolean validAddition = this.getNumRows() == other.getNumRows()
                && this.getNumCols() == other.getNumCols();
        if (!validAddition) {
            throw new IllegalArgumentException("Matrices of different dimensions are attempted to be added");
        }
        for (int rowIndex = 0; rowIndex < this.numRows; rowIndex++) {
            for (int colIndex = 0; colIndex < this.numCols; colIndex++) {
                this.arr[rowIndex][colIndex] += other.get(rowIndex, colIndex);
            }
        }

    }
    public void add(int rowIndex, int colIndex, double value) {
        arr[rowIndex][colIndex] += value;
    }public void divide(int rowIndex, int colIndex, double value) {
        arr[rowIndex][colIndex] /= value;
    }
    public static Matrix sum(Matrix m_1, Matrix m_2) {
        boolean validAddition = m_1.getNumRows() == m_2.getNumRows()
                && m_1.getNumCols() == m_2.getNumCols();
        if (!validAddition) {
            throw new IllegalArgumentException("Matrices of different dimensions are attempted to be added");
        }
        Matrix newMatrix = new Matrix(m_1.getNumRows(), m_1.getNumCols());
        for (int rowIndex = 0; rowIndex < m_1.getNumRows(); rowIndex++) {
            for (int colIndex = 0; colIndex < m_1.getNumCols(); colIndex++) {
                newMatrix.set(rowIndex, colIndex, m_1.get(rowIndex, colIndex) + m_2.get(rowIndex, colIndex));
            }
        }
        return newMatrix;
    }
    public static Matrix difference(Matrix m_1, Matrix m_2) {
        return sum(m_1, m_2.neg());
    }
    public Matrix neg() {
        Matrix newMatrix = new Matrix(numRows, numCols);
        for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
            for (int colIndex = 0; colIndex < numCols; colIndex++) {
                newMatrix.set(rowIndex, colIndex, -1 * this.get(rowIndex, colIndex));
            }
        }
        return newMatrix;
    }
    public void subtract(Matrix other) {
        this.add(other.neg());
    }
    public Matrix dot(Matrix other) {
        boolean isValidMultiplication = this.getNumCols() == other.getNumRows();
        //ystem.out.printf("%dx%d * %dx%d\n", this.numRows, this.numCols, other.getNumRows(), other.getNumCols());
        if (!isValidMultiplication) {
            throw new IllegalArgumentException("Matrices of invalid dimensions are attempted to be dot-product ed");
        }
        Matrix newMatrix = new Matrix(this.getNumRows(), other.getNumCols());
        for (int rowIndex = 0; rowIndex < newMatrix.getNumRows(); rowIndex++) {
            for (int colIndex = 0; colIndex < newMatrix.getNumCols(); colIndex++) {
                newMatrix.set(rowIndex, colIndex, Utils.dot(this.getRow(rowIndex), other.getCol(colIndex)));
            }
        }
        return newMatrix;
    }
    public Matrix dot(double value) {
        Matrix newMatrix = new Matrix(this.getNumRows(), this.getNumCols());
        for (int rowIndex = 0; rowIndex < this.getNumRows(); rowIndex++) {
            for (int colIndex = 0; colIndex < this.getNumCols(); colIndex++) {
                newMatrix.set(rowIndex, colIndex, this.arr[rowIndex][colIndex] * value);
            }
        }
        return newMatrix;
    }
    public Matrix hadamardProduct(Matrix other) {
        boolean isValidMultiplication = this.getNumRows() == other.getNumRows() && this.getNumCols() == other.getNumCols();
        if (!isValidMultiplication) {
            throw new IllegalArgumentException("Matrices of invalid dimensions are attempted to be hadamard multiplied");
        }
        Matrix newMatrix = new Matrix(this.getNumRows(), this.getNumCols());
        for (int rowIndex = 0; rowIndex < newMatrix.getNumRows(); rowIndex++) {
            for (int colIndex = 0; colIndex < newMatrix.getNumCols(); colIndex++) {
                newMatrix.set(rowIndex, colIndex, this.get(rowIndex, colIndex) * other.get(rowIndex, colIndex));
            }
        }
        return newMatrix;
    }
    public int getNumRows() {
        return this.numRows;
    }
    public int getNumCols() {
        return this.numCols;
    }
    public double[] getRow(int rowIndex) {
        return this.arr[rowIndex];
    }
    public double[] getCol(int colIndex) {
        double[] col = new double[numRows];
        int index = 0;
        for (double[] row : arr) {
            col[index++] = row[colIndex];
        }
        return col;
    }
    public void apply(Function<Double, Double> func) {
        for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
            for (int colIndex = 0; colIndex < numCols; colIndex++) {
                this.arr[rowIndex][colIndex] = func.apply(this.arr[rowIndex][colIndex]);
            }
        }
    }
    public Matrix transpose() {
        Matrix newMatrix = new Matrix(this.getNumCols(), this.getNumRows());
        for (int row = 0; row < this.getNumRows(); row++) {
            for (int col = 0; col < this.getNumCols(); col++) {
                newMatrix.set(col, row, this.arr[row][col]);
            }
        }
        return newMatrix;
    }
    public void fillWithRand(double rangeStart, double rangeEnd) {
        for (int rowIndex = 0; rowIndex < this.getNumRows(); rowIndex++) {
            for (int colIndex = 0; colIndex < this.getNumCols(); colIndex++) {
                Random random = new Random();
                this.set(rowIndex, colIndex, random.nextDouble(rangeStart, rangeEnd));
            }
        }
    }
    public static Matrix fromOneDimensionalArray(double[] array) {
        Matrix newMatrix = new Matrix(1, array.length);
        for (int colIndex = 0; colIndex < array.length; colIndex++) {
            newMatrix.set(0, colIndex, array[colIndex]);
        }
        return newMatrix;
    }

    public static double[] toOneDimensionalArray(Matrix matrix) {
        if (matrix.getNumCols() != 1 && matrix.getNumRows() != 1) {
            throw new IllegalArgumentException("toOneDimensionalArray called on matrix with multiple dimensions");
        }
        double[] oneD = new double[matrix.getNumRows()];
        if (matrix.getNumRows() == 1 && matrix.getNumCols() == 1) {
            return new double[]{matrix.get(0, 0)};
        }
        else if (matrix.getNumRows() == 1) {
            return matrix.arr[0];
        }
        else {
            int index = 0;
            for (double[] row : matrix.toArray()) {
                oneD[index++] = row[0];
            }
            return oneD;
        }

    }
    public static Matrix toDiagonalMatrix(double[] array) {
        Matrix newMatrix = new Matrix(array.length, array.length);
        for (int index = 0; index < array.length; index++) {
            newMatrix.set(index, index, array[index]);
        }
        return newMatrix;
    }
    public String toString() {
        return this.numRows + "x" + this.numCols;
    }
    public String toStringValues() {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < this.numRows; row++) {
            for (int col = 0; col < this.numCols; col++) {
                sb.append(String.format("%.4f ", arr[row][col]));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

}
