
package vml;

import cern.colt.function.*;
import cern.colt.matrix.*;
import cern.colt.matrix.linalg.Algebra;
import java.util.Random;

/**
 * Class for doing a range of matrix and vector operations.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class op 
{
    /**
     * Randomizer to use
     */
    public static Random rnd = new Random(2);
    
    /********************************
     * 
     * Create matrix or vector
     * 
     ********************************/
    
    /**
     * Creates a new matrix with zero values.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @return The matrix
     */
    public static DoubleMatrix2D matrix_zeros(int rows, int cols)
    {
        return DoubleFactory2D.dense.make(rows, cols);
    }
    
    public static DoubleMatrix2D matrix_rnd(int rows, int cols)
    {
        //return DoubleFactory2D.dense.random(rows, cols);
        return matrix_rnd(rows, cols, 1.0);
    }
    /**
     * Creates a new matrix with random values.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param scale Scale of the random values
     * @return The matrix
     */
    public static DoubleMatrix2D matrix_rnd(int rows, int cols, double scale)
    {
        DoubleMatrix2D m = matrix_zeros(rows, cols);
        for (int r = 0; r < m.rows(); r++)
        {
            for (int c = 0; c < m.columns(); c++)
            {
                m.set(r, c, rnd.nextDouble() * scale);
            }
        }
        return m;
    }
    
    /**
     * Creates a new matrix with the specified values.
     * 
     * @param values Matrix values
     * @return The matrix
     */
    public static DoubleMatrix2D matrix(double[][] values)
    {
        return DoubleFactory2D.dense.make(values);
    }
    
    /**
     * Creates a new vector with zero values.
     * 
     * @param size Size of vector
     * @return The vector
     */
    public static DoubleMatrix1D vector_zeros(int size)
    {
        return DoubleFactory1D.dense.make(size);
    }
    
    /**
     * Creates a new vector with random values.
     * 
     * @param size Size of vector
     * @return The vector
     */
    public static DoubleMatrix1D vector_rnd(int size)
    {
        //return DoubleFactory1D.dense.random(size);
        return vector_rnd(size, 1.0);
    }
    
    /**
     * Creates a new vector with random values.
     * 
     * @param size Size of vector
     * @param scale Scale of the random values
     * @return The vector
     */
    public static DoubleMatrix1D vector_rnd(int size, double scale)
    {
        //return DoubleFactory1D.dense.random(size);
        DoubleMatrix1D m = vector_zeros(size);
        for (int c = 0; c < m.size(); c++)
        {
            m.set(c, rnd.nextDouble() * scale);
        }
        return m;
    }
    
    /**
     * Creates a new vector with the specified values.
     * 
     * @param values Vector values
     * @return The vector
     */
    public static DoubleMatrix1D vector(double[] values)
    {
        return DoubleFactory1D.dense.make(values);
    }
    
    /********************************
     * 
     * View matrix
     * 
     ********************************/
    
    /**
     * Returns a row in a matrix as a vector.
     * 
     * @param m The matrix
     * @param r The row
     * @return The row as a matrix
     */
    public static DoubleMatrix1D row(DoubleMatrix2D m, int r)
    {
        return m.viewRow(r);
    }
    
    /**
     * Returns a column in a matrix as a vector.
     * 
     * @param m The matrix
     * @param c The column
     * @return The column as a matrix
     */
    public static DoubleMatrix1D column(DoubleMatrix2D m, int c)
    {
        return m.viewColumn(c);
    }
    
    /**
     * Returns the index of the highest value in a vector.
     * 
     * @param v The vector
     * @return Index of the highest value
     */
    public static int argmax(DoubleMatrix1D v)
    {
        int best_i = 0;
        double best_v = v.get(best_i);
        
        for (int i = 1; i < v.size(); i++)
        {
            double cv = v.get(i);
            if (cv > best_v)
            {
                best_v = cv;
                best_i = i;
            }
        }
        
        return best_i;
    }
    
    /********************************
     * 
     * Linear algerbra operations
     * 
     ********************************/
    
    //Used for matrix calculations
    private static Algebra alg = new Algebra();
    
    /**
     * Multiplies two matrixes.
     * 
     * @param a First matrix
     * @param b Second matrix
     * @return Result matrix
     */
    public static DoubleMatrix2D mul(DoubleMatrix2D a, DoubleMatrix2D b)
    {
        return alg.mult(a, b);
    }
    
    /**
     * Divide all values in a vector by a constant.
     * 
     * @param m The vector
     * @param scale The constant
     * @return Result vector
     */
    public static DoubleMatrix1D divide(DoubleMatrix1D m, double scale)
    {
        DoubleFunction f = (double v) -> v / scale;
        m.assign(f);
        return m;
    }
    
    /**
     * Calculates exponents of all values in a vector.
     * 
     * @param m The vector
     * @return Result vector
     */
    public static DoubleMatrix1D exp(DoubleMatrix1D m)
    {
        DoubleFunction f = (double v) -> Math.pow(Math.E, v);
        m.assign(f);
        return m;
    }
    
    /**
     * Shift all values in a matrix so that highest value in each column is 0.
     * 
     * @param m The matrix
     * @return Result matrix
     */
    public static DoubleMatrix2D shift_columns(DoubleMatrix2D m)
    {
        for (int c = 0; c < m.columns(); c++)
        {
            //Calculate max
            double max = 0;
            for (int r = 0; r < m.rows(); r++)
            {
                if (m.get(r, c) > max) max = m.get(r, c);
            }
            
            //Shift values
            for (int r = 0; r < m.rows(); r++)
            {
                m.set(r, c, m.get(r, c) - max);
            }
        }
        return m;
    }
    
    /**
     * Calculates exponents of all values in a matrix.
     * 
     * @param m The matrix
     * @return Result matrix
     */
    public static DoubleMatrix2D exp(DoubleMatrix2D m)
    {
        DoubleFunction f = (double v) -> Math.pow(Math.E, v);
        m.assign(f);
        return m;
    }
    
    /**
     * Normalizes a matrix (each value is divided by the sum of the column).
     * 
     * @param m The matrix
     * @return Result matrix
     */
    public static DoubleMatrix2D average_columns(DoubleMatrix2D m)
    {
        for (int c = 0; c < m.columns(); c++)
        {
            //Calculate sum
            double sum = 0;
            for (int r = 0; r < m.rows(); r++)
            {
                sum += m.get(r, c);
            }
            
            //Normalize values
            for (int r = 0; r < m.rows(); r++)
            {
                m.set(r, c, m.get(r, c) / sum);
            }
        }
        return m;
    }
    
    /**
     * Calculates the sum of all values in a vector.
     * 
     * @param m The vector
     * @return Sum of all values
     */
    public static double sum(DoubleMatrix1D m)
    {
        return m.zSum();
    }
    
    /**
     * Divide all values in a matrix by a constant.
     * 
     * @param m The matrix
     * @param scale The constant
     */
    public static void divide(DoubleMatrix2D m, double scale)
    {
        DoubleFunction f = (double v) -> v / scale;
        m.assign(f);
    }
    
    /**
     * Multiplies all values in a matrix by a constant.
     * 
     * @param m The matrix
     * @param scale The constant
     */
    public static void scale(DoubleMatrix2D m, double scale)
    {
        DoubleFunction f = (double v) -> v * scale;
        m.assign(f);
    }
    
    /**
     * Multiplies all values in a vector by a constant.
     * 
     * @param m The vector
     * @param scale The constant
     */
    public static void scale(DoubleMatrix1D m, double scale)
    {
        DoubleFunction f = (double v) -> v * scale;
        m.assign(f);
    }
    
    /**
     * Adds a matrix and a vector.
     * 
     * @param a The matrix
     * @param b The vector
     * @return Result matrix
     */
    public static DoubleMatrix2D add(DoubleMatrix2D a, DoubleMatrix1D b)
    {
        for (int c = 0; c < a.columns(); c++)
        {
            for (int r = 0; r < a.rows(); r++)
            {
                a.set(r, c, a.get(r, c) + b.get(r));
            }
        }
        return a;
    }
    
    /**
     * Adds two matrixes.
     * 
     * @param a First matrix
     * @param b Second matrix
     */
    public static void add(DoubleMatrix2D a, DoubleMatrix2D b)
    {
        DoubleDoubleFunction f = (double v1, double v2) -> v1 + v2;
        a.assign(b, f);
    }
    
    /**
     * Adds a vector to a row in a matrix.
     * 
     * @param m The matrix
     * @param v The vector to append
     * @param r Row number to append to
     */
    public static void add(DoubleMatrix2D m, DoubleMatrix1D v, int r)
    {
        add(m, v, r, 1.0);
    }
    
    /**
     * Adds a constant to an index in a vectot.
     * 
     * @param m The vector
     * @param i Index to append to
     * @param v The value to add
     */
    public static void add(DoubleMatrix1D m, int i, double v)
    {
        double old = m.get(i);
        m.set(i, old + v);
    }
    
    /**
     * Adds a scaled vector to a row in a matrix.
     * 
     * @param m The matrix
     * @param v The vector to append
     * @param r Row number to append to
     * @param scale Constant to scale the vector
     */
    public static void add(DoubleMatrix2D m, DoubleMatrix1D v, int r, double scale)
    {
        for (int x = 0; x < v.size(); x++)
        {
            double old = m.get(r, x);
            m.set(r, x, old + v.get(x) * scale);
        }
    }
    
    /**
     * Adds a scaled matrix b to a vector a.
     * 
     * @param a Matrix a
     * @param b Matrix b
     * @param scale Constant to scale matrix b with
     */
    public static void add(DoubleMatrix2D a, DoubleMatrix2D b, double scale)
    {
        for (int r = 0; r < a.rows(); r++)
        {
            for (int c = 0; c < a.columns(); c++)
            {
                double old = a.get(r, c);
                a.set(r, c, old + b.get(r, c) * scale);
            }
        }
    }
    
    /**
     * Sums each row in a matrix and stores the sums in a vector.
     * 
     * @param m The matrix
     * @return Result vector
     */
    public static DoubleMatrix1D sum_rows(DoubleMatrix2D m)
    {
        DoubleMatrix1D v = op.vector_zeros(m.rows());
        
        for (int r = 0; r < m.rows(); r++)
        {
            double sum = 0;
            for (int c = 0; c < m.columns(); c++)
            {
                sum += m.get(r, c);
            }
            v.set(r, sum);
        }
        
        return v;
    }
    
    /**
     * Performs the max operation for all values in a vector.
     * 
     * @param m The vector
     * @param threshold Threshold value for max operation
     */
    public static void max(DoubleMatrix1D m, double threshold)
    {
        DoubleFunction f = (double v) -> Math.max(threshold, v);
        m.assign(f);
    }
    
    /**
     * Performs the max operation for all values in a matrix.
     * 
     * @param m The matrix
     * @param threshold Threshold value for max operation
     */
    public static void max(DoubleMatrix2D m, double threshold)
    {
        DoubleFunction f = (double v) -> Math.max(threshold, v);
        m.assign(f);
    }
    
    /**
     * Creates the transpose of a matrix.
     * 
     * @param m The matrix
     * @return The transpose of the matrix
     */
    public static DoubleMatrix2D transpose(DoubleMatrix2D m)
    {
        return alg.transpose(m);
    }
    
    /**
     * Calculates the L2 (squared Euclidean) distance between two vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return L2 distance
     */
    public static double L2_dist(DoubleMatrix1D a, DoubleMatrix1D b)
    {
        double sqdist = 0;
        for (int i = 0; i < a.size(); i++)
        {
            sqdist += Math.pow(a.get(i) - b.get(i), 2);
        }
        return sqdist;
    }
    
    /**
     * Calculates the L1 distance (absolute difference) between two vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return L1 distance
     */
    public static double L1_dist(DoubleMatrix1D a, DoubleMatrix1D b)
    {
        double sqdist = 0;
        for (int i = 0; i < a.size(); i++)
        {
            sqdist += Math.abs(a.get(i) - b.get(i));
        }
        return sqdist;
    }
}
