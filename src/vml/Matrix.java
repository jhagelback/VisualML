
package vml;

import java.text.DecimalFormat;
import java.util.Random;
import java.util.stream.*;

/**
 * Representation of a matrix.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Matrix 
{
    protected double[][] v;
    private int rows;
    private int cols;
    
    //Output formatting
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Creates a new matrix with zeros.
     * 
     * @param r Number of rows
     * @param c Number of columns
     * @return A matrix
     */
    public static Matrix zeros(int r, int c)
    {
        double[][] v = new double[r][c];
        return new Matrix(v);
    }
    
    /**
     * Creates a new matrix with random values.
     * 
     * @param r Number of rows
     * @param c Number of columns
     * @param scale Scale of the random values (default is 0 to 1)
     * @return A matrix
     */
    public static Matrix random(int r, int c, double scale)
    {
        return random(r, c, scale, new Random());
    }
    
    /**
     * Creates a new matrix with random values.
     * 
     * @param r Number of rows
     * @param c Number of columns
     * @param scale Scale of the random values (default is 0 to 1)
     * @param rnd Randomizer
     * @return A matrix
     */
    public static Matrix random(int r, int c, double scale, Random rnd)
    {
        //Generate random double values between: 0 ... 1
        double min = 1000;
        double max = -1000;
        double[][] v = new double[r][c];
        for (int a = 0; a < r; a++)
        {
            for (int b = 0; b < c; b++)
            {
                v[a][b] = rnd.nextDouble();
                if (v[a][b] < min) min = v[a][b];
                if (v[a][b] > max) max = v[a][b];
            }
        }
        
        //Normalize values between: -scale ... scale
        double[][] sv = new double[r][c];
        for (int a = 0; a < r; a++)
        {
            for (int b = 0; b < c; b++)
            {
                sv[a][b] = (v[a][b] - min) / (max - min) * scale * 2 - scale;
            }
        }
        
        //Return normalized matrix
        return new Matrix(sv);
    }
    
    /**
     * Creates a new matrix with values sampled from a (normal) Gaussian distribution
     * with standard deviation of 2/Sqrt(noInputs).
     * 
     *  @param r Number of rows
     * @param c Number of columns (no inputs)
     * @param rnd Randomizer
     * @return A matrix
     */
    public static Matrix randomNormal(int r, int c, Random rnd)
    {
        //TODO: This shall actually be
        //sqrt(2) * sqrt(2/(noIn+noOut))
        //double stddev = Math.sqrt(2.0) * Math.sqrt(2.0/(r+c));
        
        //Desired standard deviation: 2 / sqrt(noInputs)
        double stddev = 2.0 / Math.sqrt(c);
        
        //Generate random double values between: 0 ... 1
        double[][] v = new double[r][c];
        for (int a = 0; a < r; a++)
        {
            for (int b = 0; b < c; b++)
            {
                v[a][b] = rnd.nextGaussian() * stddev;
            }
        }
        
        //Return matrix
        return new Matrix(v);
    }
    
    /**
     * Creates a new matrix.
     * 
     * @param v Values
     */
    public Matrix(double[][] v)
    {
        this.v = v;
        rows = v.length;
        cols = v[0].length;
    }
    
    /**
     * Returns a value in the matrix.
     * 
     * @param r Row
     * @param c Column
     * @return The value
     */
    public double get(int r, int c)
    {
        return v[r][c];
    }
    
    /**
     * Sets a value in the matrix.
     * 
     * @param r Row
     * @param c Column
     * @param value The value
     */
    public void set(int r, int c, double value)
    {
        v[r][c] = value;
    }
    
    /**
     * Returns the number of rows in the matrix.
     * 
     * @return Number of rows
     */
    public int rows()
    {
        return rows;
    }
    
    /**
     * Returns the number of columns in the matrix.
     * 
     * @return Number of columns
     */
    public int columns()
    {
        return cols;
    }
    
    /**
     * Returns a column in this matrix as a vector.
     * 
     * @param c Column number
     * @return The column as vector
     */
    public Vector getColumn(int c)
    {
        double[] nv = new double[rows()];
        
        for (int r = 0; r < rows(); r++)
        {
            nv[r] = v[r][c];
        }
        
        return new Vector(nv);
    }
    
    /**
     * Adds a value to a position in the matrix.
     * 
     * @param r Row
     * @param c Column
     * @param value The value to add
     */
    public void add(int r, int c, double value)
    {
        v[r][c] += value;
    }
    
    /**
     * Calculates activation (w*x+b) for a weights matrix and an input vector.
     * Input matrix and vectors are not modified.
     * 
     * @param w Weights matrix
     * @param x Input vector
     * @param b Bias vector
     * @return Activation vector
     * @throws ArithmeticException If unable to calculate the activation vector 
     */
    public static Vector activation(Matrix w, Vector x, Vector b) throws ArithmeticException
    {
        //Error checks
        if (w.columns() != x.size())
        {
            throw new ArithmeticException("Number of columns in weights matrix does not match size of input vector");
        }
        if (w.rows() != b.size())
        {
            throw new ArithmeticException("Number of rows in weights matrix does not match size of bias vector");
        }
        
        //Activation vector
        double[] nv = new double[w.rows()];
        
        IntStream.range(0, w.rows()).parallel().forEach(r -> {
            //Multiply the row in weight matrix with the input vector
            for (int c = 0; c < w.columns(); c++)
            {
                nv[r] += w.get(r, c) * x.get(c);
            }
            //Add bias
            nv[r] += b.get(r);
        });
        
        return new Vector(nv);
    }
    
    /**
     * Calculates activation (w*x+b) for a weights matrix and an input matrix.
     * Input matrices and vector are not modified.
     * 
     * @param w Weights matrix
     * @param x Input matrix
     * @param b Bias vector
     * @return Activation vector
     * @throws ArithmeticException If unable to calculate the activation vector 
     */
    public static Matrix activation(Matrix w, Matrix x, Vector b) throws ArithmeticException
    {
        //Error checks
        if (w.columns() != x.rows())
        {
            throw new ArithmeticException("Number of columns in weights matrix does not match rows of input matrix");
        }
        if (w.rows() != b.size())
        {
            throw new ArithmeticException("Number of rows in weights matrix does not match size of bias vector");
        }
        
        //Activation vector
        double[][] nv = new double[w.rows()][x.columns()];
        
        IntStream.range(0, x.columns()).parallel().forEach(nc -> {
            for (int r = 0; r < w.rows(); r++)
            {
                //Multiply the row in weight matrix with the input vector
                for (int c = 0; c < w.columns(); c++)
                {
                    nv[r][nc] += w.v[r][c] * x.v[c][nc];
                }
                //Add bias
                nv[r][nc] += b.v[r];
            }
        });
        
        return new Matrix(nv);
    }
    
    /**
     * Calculates the product of the transpose of w and d.
     * 
     * @param w The matrix to transpose
     * @param d The other matrix
     * @return Result matrix
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Matrix transpose_mul(Matrix w, Matrix d) throws ArithmeticException
    {
        //Error checks
        if (w.rows() != d.rows())
        {
            throw new ArithmeticException("Number of rows in first matrix does not match rows of second matrix");
        }
        
        //Result matrix
        double[][] nv = new double[w.columns()][d.columns()];
        
        IntStream.range(0, d.columns()).parallel().forEach(nc -> {
            for (int r = 0; r < w.columns(); r++)
            {
                for (int c = 0; c < w.rows(); c++)
                {
                    nv[r][nc] += w.v[c][r] * d.v[c][nc]; //Exchange rows with cols in w to get transpose
                }
            }
        });
        
        return new Matrix(nv);
    }
    
    /**
     * Multiplies a matrix with the transpose of another matrix.
     * 
     * @param d First matrix
     * @param x Second matrix
     * @return Result matrix
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Matrix mul_transpose(Matrix d, Matrix x) throws ArithmeticException
    {
        //Error checks
        if (d.columns() != x.columns())
        {
            throw new ArithmeticException("Number of columns in first matrix does not match columns of second matrix");
        }
        
        //Result matrix
        double[][] nv = new double[d.rows()][x.rows()];
        
        IntStream.range(0, x.rows()).parallel().forEach(nc -> {
            for (int r = 0; r < d.rows(); r++)
            {
                for (int c = 0; c < d.columns(); c++)
                {
                    nv[r][nc] += d.v[r][c] * x.v[nc][c]; //Exchange rows with cols in x to get transpose
                }
            }
        });
        
        return new Matrix(nv);
    }
    
    /**
     * Multiplies two matrices.
     * Input matrices and vector are not modified.
     * 
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Result matrix
     * @throws ArithmeticException If unable to calculate the product 
     */
    public static Matrix mul(Matrix m1, Matrix m2) throws ArithmeticException
    {
        //Error checks
        if (m1.columns() != m2.rows())
        {
            throw new ArithmeticException("Number of columns in first matrix does not match rows of second matrix");
        }
        
        //Result matrix
        double[][] nv = new double[m1.rows()][m2.columns()];
        
        IntStream.range(0, m2.columns()).parallel().forEach(nc -> {
            for (int r = 0; r < m1.rows(); r++)
            {
                //Multiply the row in weight matrix with the input vector
                for (int c = 0; c < m1.columns(); c++)
                {
                    nv[r][nc] += m1.v[r][c] * m2.v[c][nc];
                }
            }
        });
        
        return new Matrix(nv);
    }
    
    /**
     * Multiplies a matrix with a vector.
     * 
     * @param w The matrix
     * @param x The vector
     * @return Result vector
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Vector mul(Matrix w, Vector x) throws ArithmeticException
    {
        //Error checks
        if (w.columns() != x.size())
        {
            throw new ArithmeticException("Number of columns in matrix does not match size of vector");
        }
        
        //Activation vector
        double[] nv = new double[w.rows()];
        
        IntStream.range(0, w.rows()).parallel().forEach(r -> {
            //Multiply the row in the matrix with the vector
            for (int c = 0; c < w.columns(); c++)
            {
                nv[r] += w.get(r, c) * x.get(c);
            }
        });
        
        return new Vector(nv);
    }
    
    /**
     * Multiplies a transposed vector with a matrix.
     * 
     * @param x The vector to transpose
     * @param w The matrix
     * @return Result vector
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Vector transpose_mul(Vector x, Matrix w) throws ArithmeticException
    {
        //Error checks
        if (w.rows() != x.size())
        {
            throw new ArithmeticException("Number of rows in matrix does not match size of vector");
        }
        
        //Activation vector
        double[] nv = new double[w.rows()];
        
        IntStream.range(0, w.rows()).parallel().forEach(r -> {
            //Multiply the row vector with each column in the vector
            for (int c = 0; c < w.columns(); c++)
            {
                nv[r] += w.get(r, c) * x.get(c);
            }
        });
        
        return new Vector(nv);
    }
    
    /**
     * Calculates the transpose of a matrix.
     * 
     * @param m The matrix
     * @return Transpose of the matrix
     */
    public static Matrix transpose(Matrix m)
    {
        double[][] v = new double[m.columns()][m.rows()];
        for (int r = 0; r < m.rows(); r++)
        {
            for (int c = 0; c < m.columns(); c++)
            {
                v[c][r] = m.v[r][c];
            }
        }
        
        return new Matrix(v);
    }
    
    /**
     * Calculates the L2 norm (sum of all squared values) for this matrix.
     * 
     * @return L2 norm
     */
    public double L2_norm()
    {
        double norm = 0;
        
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                norm += get(r,c) * get(r,c);
            }
        }
        
        return norm;
    }
    
    /**
     * For each column, subtracts the values by the max value of that column resulting
     * in all values being less than or equal to zero.
     * 
     * @return Result matrix
     */
    public Matrix shift_columns()
    {
        IntStream.range(0, columns()).parallel().forEach(c -> {
            //Calculate max
            double max = 0;
            for (int r = 0; r < rows(); r++)
            {
                if (v[r][c] > max) max = v[r][c];
            }
            
            //Shift values
            for (int r = 0; r < rows(); r++)
            {
                v[r][c] -= max;
            }
        });
        
        return this;
    }
    
    /**
     * Normalizes each column in the vector so the sum of each column is 1.
     */
    public void normalize()
    {
        IntStream.range(0, columns()).parallel().forEach(c -> {
            //Calculate sum
            double sum = 0;
            for (int r = 0; r < rows(); r++)
            {
                sum += v[r][c];
            }
            
            //Normalize values
            for (int r = 0; r < rows(); r++)
            {
                v[r][c] /= sum;
            }
        });
    }
    
    /**
     * Calculates the dscores matrix from normalize log probabilities vector.
     * 
     * @param y Correct class labels
     * 
     * @return Dscores matrix
     */
    public Matrix calc_dscores(Vector y)
    {
        IntStream.range(0, columns()).parallel().forEach(c -> {
            //Find correct label for this training example
            int corr_index = (int)y.get(c);
            //Subtract the column value by 1
            v[corr_index][c] -= 1.0;
            //Divide by number of training examples
            for (int r = 0; r < rows(); r++)
            {
                v[r][c] /= y.size();
            }
        });
        
        return this;
    }
    
    /**
     * Calculates the Softmax cross-entropy loss vector for this matrix.
     * 
     * @param y Correct class labels
     * @return Loss vector
     */
    public Vector calc_loss(Vector y)
    {
        //Loss vector values
        double[] L = new double[y.size()];
        
        for (int c = 0; c < columns(); c++)
        {
            //Find correct class score for this training example
            double class_score = v[(int)y.get(c)][c];
            //Calculate loss
            double Li = -1.0 * Math.log(class_score) / Math.log(Math.E);
            L[c] = Li;
        }
        
        return new Vector(L);
    }
    
    /**
     * Calculates E^v for all values in the matrix.
     * 
     * @return Result matrix
     */
    public Matrix exp()
    {
        IntStream.range(0, columns()).parallel().forEach(c -> {
            for (int r = 0; r < rows(); r++)
            {
                v[r][c] = Math.pow(Math.E, v[r][c]);
            }
        });
        
        return this;
    }
    
    /**
     * Returns the sum of a specified row in this matrix.
     * 
     * @param r The rum
     * @return Sum of the row
     */
    public double sum_row(int r)
    {
        double sum = 0;
        for (int c = 0; c < columns(); c++)
        {
            sum += v[r][c];
        }
        return sum;
    }
    
    /**
     * Returns the sum of a specified column in this matrix.
     * 
     * @param c The column
     * @return Sum of the column
     */
    public double sum_col(int c)
    {
        double sum = 0;
        for (int r = 0; r < rows(); r++)
        {
            sum += v[r][c];
        }
        return sum;
    }
    
    
    /**
     * Creates a new vector with the sum of each row in this matrix.
     * 
     * @return The vector
     */
    public Vector sum_rows()
    {
        //Loss vector values
        double[] sum = new double[rows()];
        
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                sum[r] += v[r][c];
            }
        }
        
        return new Vector(sum);
    }
    
    /**
     * Adds a scaled matrix to this matrix.
     * 
     * @param m The other matrix
     * @param scale Scale
     */
    public void add(Matrix m, double scale)
    {
        //Error checks
        if (rows() != m.rows() || columns() != m.columns())
        {
            throw new ArithmeticException("Size of matrices does not match");
        }
        
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                v[r][c] += m.get(r,c) * scale;
            }
        }
    }
    
    /**
     * Updates the weights, assuming this is a weights matrix.
     * 
     * @param dW Gradients matrix
     * @param learningrate Learning rate
     */
    public void update_weights(Matrix dW, double learningrate)
    {
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                v[r][c] -= dW.get(r,c) * learningrate;
            }
        }
    }
    
    /**
     * Calculates the index for the highest value in a column.
     * 
     * @param c Column
     * @return Index of highest value
     */
    public int argmax(int c)
    {
        double high = Double.NEGATIVE_INFINITY;
        int index = -1;
        
        for (int r = 0; r < rows(); r++)
        {
            if (v[r][c] > high)
            {
                high = v[r][c];
                index = r;
            }
        }
        
        return index;
    }
    
    /**
     * Divides all values in the matrix by a constant.
     * 
     * @param cons The constant
     */
    public void div(double cons)
    {
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                v[r][c] /= cons;
            }
        }
    }
    
    /**
     * Multiplies all values in the matrix by a constant.
     * 
     * @param cons The constant
     */
    public void multiply(double cons)
    {
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                v[r][c] *= cons;
            }
        }
    }
    
    /**
     * Piece-wise subtraction of all values in the matrix by another matrix.
     * 
     * @param m The matrix to subtract
     */
    public void subtract(Matrix m)
    {
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                v[r][c] -= m.v[r][c];
            }
        }
    }
    
    /**
     * Inserts a column vector at the specified column in this matrix.
     * 
     * @param x The column vector
     * @param c Column to insert at
     */
    public void insert(Vector x, int c) throws ArithmeticException
    {
        //Error check
        if (rows() != x.size())
        {
            throw new ArithmeticException("Number of rows in the matrix does not match size of vector");
        }
        
        //Copy values from the vector to a column in the matrix
        for (int r = 0; r < rows(); r++)
        {
            v[r][c] = x.v[r];
        }
    }
    
    /**
     * Performs the max operation on all values in the matrix.
     * 
     * @param max_val Max value
     */
    public void max(double max_val)
    {
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                v[r][c] = Math.max(v[r][c], max_val);
            }
        }
    }
    
    /**
     * Adds a scaled vector to a row in this matrix.
     * 
     * @param rv The vector to append
     * @param r Row number to append to
     * @param scale Constant to scale the vector
     */
    public void addToRow(Vector rv, int r, double scale)
    {
        for (int c = 0; c < rv.size(); c++)
        {
            v[r][c] += rv.get(c) * scale;
        }
    }
    
    /**
     * Backpropagates the ReLU non-linearity into the gradients matrix. All gradient values
     * are set to 0 if the corresponding activation value is 0.
     * 
     * @param scores Activation matrix
     */
    public void backprop_relu(Matrix scores)
    {
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                //Check if activation is <= 0
                if (scores.get(r, c) <= 0)
                {
                    //Switch off
                    v[r][c] = 0;
                }
            }
        }
    }
    
    /**
     * Creates a submatrix from this matrix.
     * 
     * @param m The matrix
     * @param rows Number of rows to keep
     * @param columns Number of columns to keep
     * @return The submatrix
     */
    public static Matrix submatrix(Matrix m, int rows, int columns) throws ArithmeticException
    {
        //Error check
        if (rows < 1 || rows > m.rows() || columns < 1 || columns > m.columns())
        {
            throw new ArithmeticException("Invalid size of submatrix");
        }
        
        double[][] v = new double[rows][columns];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                v[r][c] = m.v[r][c];
            }
        }
        
        Matrix red = new Matrix(v);
        return red;
    }
    
    /**
     * Creates a copy of this matrix.
     * 
     * @return The copy of this matrix
     */
    public Matrix copy()
    {
        double[][] nv = new double[rows()][columns()];
        for (int r = 0; r < rows(); r++)
        {
            System.arraycopy(v[r], 0, nv[r], 0, columns());
        }
        return new Matrix(nv);
    }
    
    @Override
    public String toString()
    {
        String str = rows() + "x" + columns() + " matrix\n";
        for (int r = 0; r < rows(); r++)
        {
            str += "[";
            for (int c = 0; c < columns(); c++)
            {
                str += df.format(v[r][c]);
                if (c < columns() - 1)
                {
                    str += ", ";
                }
            }
            str += "]";
            if (r < rows() - 1) str += "\n";
        }
        
        return str;
    }
}
