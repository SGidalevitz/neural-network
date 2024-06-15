package Tests;

import org.junit.Assert;
import org.junit.Test;
import Main.Matrix;

import static org.junit.Assert.*;

public class MatrixTest {

    @Test
    public void testGet() {
        Matrix matrix = new Matrix(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        assertEquals(1.0, matrix.get(0, 0), 0.0);
        assertEquals(2.0, matrix.get(0, 1), 0.0);
        assertEquals(3.0, matrix.get(1, 0), 0.0);
    }

    @Test
    public void testSet() {
        Matrix matrix = new Matrix(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);

        assertEquals(1.0, matrix.get(0, 0), 0.0);
        assertEquals(2.0, matrix.get(0, 1), 0.0);
        assertEquals(3.0, matrix.get(1, 0), 0.0);
    }

    @Test
    public void testAdd() {
        Matrix matrix1 = new Matrix(2, 2);
        matrix1.set(0, 0, 1.0);
        matrix1.set(0, 1, 2.0);
        matrix1.set(1, 0, 3.0);
        matrix1.set(1, 1, 4.0);

        Matrix matrix2 = new Matrix(2, 2);
        matrix2.set(0, 0, 5.0);
        matrix2.set(0, 1, 6.0);
        matrix2.set(1, 0, 7.0);
        matrix2.set(1, 1, 8.0);

        matrix1.add(matrix2);

        assertEquals(6.0, matrix1.get(0, 0), 0.0);
        assertEquals(8.0, matrix1.get(0, 1), 0.0);
        assertEquals(10.0, matrix1.get(1, 0), 0.0);
    }

    @Test
    public void testNeg() {
        Matrix matrix = new Matrix(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        Matrix result = matrix.neg();

        assertEquals(-1.0, result.get(0, 0), 0.0);
        assertEquals(-2.0, result.get(0, 1), 0.0);
        assertEquals(-3.0, result.get(1, 0), 0.0);
    }

    @Test
    public void testSubtract() {
        Matrix matrix1 = new Matrix(2, 2);
        matrix1.set(0, 0, 1.0);
        matrix1.set(0, 1, 2.0);
        matrix1.set(1, 0, 3.0);
        matrix1.set(1, 1, 4.0);

        Matrix matrix2 = new Matrix(2, 2);
        matrix2.set(0, 0, 5.0);
        matrix2.set(0, 1, 6.0);
        matrix2.set(1, 0, 7.0);
        matrix2.set(1, 1, 8.0);

        matrix1.subtract(matrix2);

        assertEquals(-4.0, matrix1.get(0, 0), 0.0);
        assertEquals(-4.0, matrix1.get(0, 1), 0.0);
        assertEquals(-4.0, matrix1.get(1, 0), 0.0);
    }

    @Test
    public void testDot() {
        Matrix matrix1 = new Matrix(2, 2);
        matrix1.set(0, 0, 1.0);
        matrix1.set(0, 1, 2.0);
        matrix1.set(1, 0, 3.0);
        matrix1.set(1, 1, 4.0);

        Matrix matrix2 = new Matrix(2, 2);
        matrix2.set(0, 0, 5.0);
        matrix2.set(0, 1, 6.0);
        matrix2.set(1, 0, 7.0);
        matrix2.set(1, 1, 8.0);

        Matrix result = matrix1.dot(matrix2);

        Assert.assertEquals(19.0, result.get(0, 0), 0.0);
        Assert.assertEquals(22.0, result.get(0, 1), 0.0);
        Assert.assertEquals(43.0, result.get(1, 0), 0.0);
    }
}