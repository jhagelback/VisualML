
package vml;

import java.text.DecimalFormat;
import java.util.Random;
import java.util.stream.*;

/**
 * Representation of a 2D-tensor (a matrix).
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Tensor2D 
{
    protected double[][] v;
    private int rows;
    private int cols;
    
    //Output formatting
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Creates a new 2D-tensor with zeros.
     * 
     * @param r Number of rows
     * @param c Number of columns
     * @return A 2D-tensor
     */
    public static Tensor2D zeros(int r, int c)
    {
        double[][] v = new double[r][c];
        return new Tensor2D(v);
    }
    
    /**
     * Creates a new 2D-tensor with random values.
     * 
     * @param r Number of rows
     * @param c Number of columns
     * @param scale Scale of the random values (default is 0 to 1)
     * @return A 2D-tensor
     */
    public static Tensor2D random(int r, int c, double scale)
    {
        return random(r, c, scale, new Random());
    }
    
    /**
     * Creates a new 2D-tensor with random values.
     * 
     * @param r Number of rows
     * @param c Number of columns
     * @param scale Scale of the random values (default is 0 to 1)
     * @param rnd Randomizer
     * @return A 2D-tensor
     */
    public static Tensor2D random(int r, int c, double scale, Random rnd)
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
        
        //Return normalized 2D-tensor
        return new Tensor2D(sv);
    }
    
    /**
     * Creates a new 2D-tensor with values sampled from a (normal) Gaussian distribution
     * with standard deviation of: 2.0/sqrt(noInputs).
     * 
     *  @param r Number of rows
     * @param c Number of columns (no inputs)
     * @param rnd Randomizer
     * @return A 2D-tensor
     */
    public static Tensor2D randomNormal(int r, int c, Random rnd)
    {
        //Desired standard deviation:
        //2.0/sqrt(noInputs)
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
        
        //Return 2D-tensor
        return new Tensor2D(v);
    }
    
    /**
     * Creates a new 2D-tensor.
     * 
     * @param v Values
     */
    public Tensor2D(double[][] v)
    {
        this.v = v;
        rows = v.length;
        cols = v[0].length;
    }
    
    /**
     * Returns a value in the 2D-tensor.
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
     * Sets a value in the 2D-tensor.
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
     * Returns the number of rows in the 2D-tensor.
     * 
     * @return Number of rows
     */
    public int rows()
    {
        return rows;
    }
    
    /**
     * Returns the number of columns in the 2D-tensor.
     * 
     * @return Number of columns
     */
    public int columns()
    {
        return cols;
    }
    
    /**
     * Returns a column in this 2D-tensor as a 1D-tensor.
     * 
     * @param c Column number
     * @return The column as 1D-tensor
     */
    public Tensor1D getColumn(int c)
    {
        double[] nv = new double[rows()];
        
        for (int r = 0; r < rows(); r++)
        {
            nv[r] = v[r][c];
        }
        
        return new Tensor1D(nv);
    }
    
    /**
     * Adds a value to a position in the 2D-tensor.
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
     * Calculates activation (w*x+b) for a weights 2D-tensor and an input 1D-tensor.
     * Input tensors are not modified.
     * 
     * @param w Weights 2D-tensor
     * @param x Input 1D-tensor
     * @param b Bias 1D-tensor
     * @return Activation 1D-tensor
     * @throws ArithmeticException If unable to calculate the activation 1D-tensor 
     */
    public static Tensor1D activation(Tensor2D w, Tensor1D x, Tensor1D b) throws ArithmeticException
    {
        //Error checks
        if (w.columns() != x.size())
        {
            throw new ArithmeticException("Number of columns in weights tensor does not match size of input tensor");
        }
        if (w.rows() != b.size())
        {
            throw new ArithmeticException("Number of rows in weights tensor does not match size of bias tensor");
        }
        
        //Activation 1D-tensor
        double[] nv = new double[w.rows()];
        
        IntStream.range(0, w.rows()).parallel().forEach(r -> {
            //Multiply the row in weight 2D-tensor with the input 1D-tensor
            for (int c = 0; c < w.columns(); c++)
            {
                nv[r] += w.get(r, c) * x.get(c);
            }
            //Add bias
            nv[r] += b.get(r);
        });
        
        return new Tensor1D(nv);
    }
    
    /**
     * Calculates activation (w*x+b) for a weights 2D-tensor and an input 2D-tensor.
     * Input tensors are not modified.
     * 
     * @param w Weights 2D-tensor
     * @param x Input 2D-tensor
     * @param b Bias 1D-tensor
     * @return Activation 1D-tensor
     * @throws ArithmeticException If unable to calculate the activation 1D-tensor 
     */
    public static Tensor2D activation(Tensor2D w, Tensor2D x, Tensor1D b) throws ArithmeticException
    {
        //Error checks
        if (w.columns() != x.rows())
        {
            throw new ArithmeticException("Number of columns in weights tensor does not match rows of input tensor");
        }
        if (w.rows() != b.size())
        {
            throw new ArithmeticException("Number of rows in weights tensor does not match size of bias tensor");
        }
        
        //Activation 1D-tensor
        double[][] nv = new double[w.rows()][x.columns()];
        
        IntStream.range(0, x.columns()).parallel().forEach(nc -> {
            for (int r = 0; r < w.rows(); r++)
            {
                //Multiply the row in weight 2D-tensor with the input 1D-tensor
                for (int c = 0; c < w.columns(); c++)
                {
                    nv[r][nc] += w.v[r][c] * x.v[c][nc];
                }
                //Add bias
                nv[r][nc] += b.v[r];
            }
        });
        
        return new Tensor2D(nv);
    }
    
    /**
     * Calculates the product of the transpose of w and d.
     * 
     * @param w The 2D-tensor to transpose
     * @param d The other 2D-tensor
     * @return Result 2D-tensor
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Tensor2D transpose_mul(Tensor2D w, Tensor2D d) throws ArithmeticException
    {
        //Error checks
        if (w.rows() != d.rows())
        {
            throw new ArithmeticException("Number of rows in first 2D-tensor does not match rows of second 2D-tensor");
        }
        
        //Result 2D-tensor
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
        
        return new Tensor2D(nv);
    }
    
    /**
     * Multiplies a 2D-tensor with the transpose of another 2D-tensor.
     * 
     * @param d First 2D-tensor
     * @param x Second 2D-tensor
     * @return Result 2D-tensor
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Tensor2D mul_transpose(Tensor2D d, Tensor2D x) throws ArithmeticException
    {
        //Error checks
        if (d.columns() != x.columns())
        {
            throw new ArithmeticException("Number of columns in first 2D-tensor does not match columns of second 2D-tensor");
        }
        
        //Result 2D-tensor
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
        
        return new Tensor2D(nv);
    }
    
    /**
     * Multiplies two matrices.
     * Input tensors are not modified.
     * 
     * @param m1 First 2D-tensor
     * @param m2 Second 2D-tensor
     * @return Result 2D-tensor
     * @throws ArithmeticException If unable to calculate the product 
     */
    public static Tensor2D mul(Tensor2D m1, Tensor2D m2) throws ArithmeticException
    {
        //Error checks
        if (m1.columns() != m2.rows())
        {
            throw new ArithmeticException("Number of columns in first 2D-tensor does not match rows of second 2D-tensor");
        }
        
        //Result 2D-tensor
        double[][] nv = new double[m1.rows()][m2.columns()];
        
        IntStream.range(0, m2.columns()).parallel().forEach(nc -> {
            for (int r = 0; r < m1.rows(); r++)
            {
                //Multiply the row in weight 2D-tensor with the input 1D-tensor
                for (int c = 0; c < m1.columns(); c++)
                {
                    nv[r][nc] += m1.v[r][c] * m2.v[c][nc];
                }
            }
        });
        
        return new Tensor2D(nv);
    }
    
    /**
     * Multiplies a 2D-tensor with a 1D-tensor.
     * 
     * @param w The 2D-tensor
     * @param x The 1D-tensor
     * @return Result 1D-tensor
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Tensor1D mul(Tensor2D w, Tensor1D x) throws ArithmeticException
    {
        //Error checks
        if (w.columns() != x.size())
        {
            throw new ArithmeticException("Number of columns in 2D-tensor does not match size of 1D-tensor");
        }
        
        //Activation 1D-tensor
        double[] nv = new double[w.rows()];
        
        IntStream.range(0, w.rows()).parallel().forEach(r -> {
            //Multiply the row in the 2D-tensor with the 1D-tensor
            for (int c = 0; c < w.columns(); c++)
            {
                nv[r] += w.get(r, c) * x.get(c);
            }
        });
        
        return new Tensor1D(nv);
    }
    
    /**
     * Multiplies a transposed 1D-tensor with a 2D-tensor.
     * 
     * @param x The 1D-tensor to transpose
     * @param w The 2D-tensor
     * @return Result 1D-tensor
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Tensor1D transpose_mul(Tensor1D x, Tensor2D w) throws ArithmeticException
    {
        //Error checks
        if (w.rows() != x.size())
        {
            throw new ArithmeticException("Number of rows in 2D-tensor does not match size of 1D-tensor");
        }
        
        //Activation 1D-tensor
        double[] nv = new double[w.rows()];
        
        IntStream.range(0, w.rows()).parallel().forEach(r -> {
            //Multiply the row 1D-tensor with each column in the 2D-tensor
            for (int c = 0; c < w.columns(); c++)
            {
                nv[r] += w.get(r, c) * x.get(c);
            }
        });
        
        return new Tensor1D(nv);
    }
    
    /**
     * Calculates the transpose of a 2D-tensor.
     * 
     * @param m The 2D-tensor
     * @return Transpose of the 2D-tensor
     */
    public static Tensor2D transpose(Tensor2D m)
    {
        double[][] v = new double[m.columns()][m.rows()];
        for (int r = 0; r < m.rows(); r++)
        {
            for (int c = 0; c < m.columns(); c++)
            {
                v[c][r] = m.v[r][c];
            }
        }
        
        return new Tensor2D(v);
    }
    
    /**
     * Calculates the L2 norm (sum of all squared values) for this 2D-tensor.
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
     * @return Result 2D-tensor
     */
    public Tensor2D shift_columns()
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
     * Normalizes each column in the tensor so the sum of each column is 1.
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
     * Calculates the dscores 2D-tensor from normalize log probabilities 1D-tensor.
     * 
     * @param y Correct class labels
     * 
     * @return Dscores 2D-tensor
     */
    public Tensor2D calc_dscores(Tensor1D y)
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
     * Calculates the Softmax cross-entropy loss 1D-tensor for this 2D-tensor.
     * 
     * @param y Correct class labels
     * @return Loss 1D-tensor
     */
    public Tensor1D calc_loss(Tensor1D y)
    {
        //Loss 1D-tensor values
        double[] L = new double[y.size()];
        
        for (int c = 0; c < columns(); c++)
        {
            //Find correct class score for this training example
            double class_score = v[(int)y.get(c)][c];
            //Calculate loss
            double Li = -1.0 * Math.log(class_score) / Math.log(Math.E);
            L[c] = Li;
        }
        
        return new Tensor1D(L);
    }
    
    /**
     * Calculates E^v for all values in the 2D-tensor.
     * 
     * @return Result 2D-tensor
     */
    public Tensor2D exp()
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
     * Returns the sum of a specified row in this 2D-tensor.
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
     * Returns the sum of a specified column in this 2D-tensor.
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
     * Creates a new 1D-tensor with the sum of each row in this 2D-tensor.
     * 
     * @return The 1D-tensor
     */
    public Tensor1D sum_rows()
    {
        //Loss 1D-tensor values
        double[] sum = new double[rows()];
        
        for (int r = 0; r < rows(); r++)
        {
            for (int c = 0; c < columns(); c++)
            {
                sum[r] += v[r][c];
            }
        }
        
        return new Tensor1D(sum);
    }
    
    /**
     * Adds a scaled 2D-tensor to this 2D-tensor.
     * 
     * @param m The other 2D-tensor
     * @param scale Scale
     */
    public void add(Tensor2D m, double scale)
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
     * Updates the weights, assuming this is a weights 2D-tensor.
     * 
     * @param dW Gradients 2D-tensor
     * @param learningrate Learning rate
     */
    public void update_weights(Tensor2D dW, double learningrate)
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
     * Divides all values in the 2D-tensor by a constant.
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
     * Multiplies all values in the 2D-tensor by a constant.
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
     * Piece-wise subtraction of all values in the 2D-tensor by another 2D-tensor.
     * 
     * @param m The 2D-tensor to subtract
     */
    public void subtract(Tensor2D m)
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
     * Inserts a column 1D-tensor at the specified column in this 2D-tensor.
     * 
     * @param x The column 1D-tensor
     * @param c Column to insert at
     */
    public void insert(Tensor1D x, int c) throws ArithmeticException
    {
        //Error check
        if (rows() != x.size())
        {
            throw new ArithmeticException("Number of rows in the 2D-tensor does not match size of 1D-tensor");
        }
        
        //Copy values from the 1D-tensor to a column in the 2D-tensor
        for (int r = 0; r < rows(); r++)
        {
            v[r][c] = x.v[r];
        }
    }
    
    /**
     * Performs the max operation on all values in the 2D-tensor.
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
     * Adds a scaled 1D-tensor to a row in this 2D-tensor.
     * 
     * @param rv The 1D-tensor to append
     * @param r Row number to append to
     * @param scale Constant to scale the 1D-tensor
     */
    public void addToRow(Tensor1D rv, int r, double scale)
    {
        for (int c = 0; c < rv.size(); c++)
        {
            v[r][c] += rv.get(c) * scale;
        }
    }
    
    /**
     * Backpropagates the ReLU non-linearity into the gradients 2D-tensor. All gradient values
     * are set to 0 if the corresponding activation value is 0.
     * 
     * @param scores Activation 2D-tensor
     */
    public void backprop_relu(Tensor2D scores)
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
     * Creates a sub 2D-tensor from this 2D-tensor.
     * 
     * @param m The 2D-tensor
     * @param rows Number of rows to keep
     * @param columns Number of columns to keep
     * @return The sub 2D-tensor
     */
    public static Tensor2D sub(Tensor2D m, int rows, int columns) throws ArithmeticException
    {
        //Error check
        if (rows < 1 || rows > m.rows() || columns < 1 || columns > m.columns())
        {
            throw new ArithmeticException("Invalid size of sub 2D-tensor");
        }
        
        double[][] v = new double[rows][columns];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                v[r][c] = m.v[r][c];
            }
        }
        
        Tensor2D red = new Tensor2D(v);
        return red;
    }
    
    /**
     * Creates a copy of this 2D-tensor.
     * 
     * @return The copy of this 2D-tensor
     */
    public Tensor2D copy()
    {
        double[][] nv = new double[rows()][columns()];
        for (int r = 0; r < rows(); r++)
        {
            System.arraycopy(v[r], 0, nv[r], 0, columns());
        }
        return new Tensor2D(nv);
    }
    
    @Override
    public String toString()
    {
        String str = rows() + "x" + columns() + " 2D-tensor\n";
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
