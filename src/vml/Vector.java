
package vml;

import java.text.DecimalFormat;
import java.util.Random;

/**
 * Representation of a vector.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Vector 
{
    protected double[] v;
    
    //Output formatting
    private DecimalFormat df = new DecimalFormat("0.000"); 
    
    /**
     * Creates a new vector with zeros.
     * 
     * @param s Size of the vector
     * @return A vector
     */
    public static Vector zeros(int s)
    {
        double[] v = new double[s];
        return new Vector(v);
    }
    
    /**
     * Creates a new vector with random values.
     * 
     * @param s Size of vector
     * @param scale Scale of the random values (default is 0 to 1)
     * @return A vector
     */
    public static Vector random(int s, double scale)
    {
        return random(s, scale, new Random());
    }
    
    /**
     * Creates a new vector with random values.
     * 
     * @param s Size of vector
     * @param scale Scale of the random values (default is 0 to 1)
     * @param rnd Randomizer
     * @return A matrix
     */
    public static Vector random(int s, double scale, Random rnd)
    {
        //Generate random double values between: 0 ... 1
        double min = 1000;
        double max = -1000;
        double[] v = new double[s];
        for (int a = 0; a < s; a++)
        {
            v[a] = rnd.nextDouble() * scale;
            if (v[a] < min) min = v[a];
            if (v[a] > max) max = v[a];
        }
        
        //Normalize values between: -scale ... scale
        double[] sv = new double[s];
        for (int a = 0; a < s; a++)
        {
            sv[a] = (v[a] - min) / (max - min) * scale * 2 - scale;
        }
        
        //Return normalized vector
        return new Vector(sv);
    }
    
    /**
     * Creates a random, normalized unit vector.
     * 
     * @param s Size of vector
     * @return Random normalized unit vector
     */
    public static Vector random_norm(int s)
    {
        Random rnd = new Random();
        //Generate random double values between: 0 ... 1
        double[] v = new double[s];
        double sqSum = 0;
        for (int a = 0; a < s; a++)
        {
            v[a] = rnd.nextDouble();
            sqSum += Math.pow(v[a], 2);
        }
        sqSum = Math.sqrt(sqSum);
        
        //Normalize values
        double[] sv = new double[s];
        for (int a = 0; a < s; a++)
        {
            sv[a] = v[a] / sqSum;
        }
        
        //Return normalized vector
        return new Vector(sv);
    }
    
    /**
     * Creates a new vector.
     * 
     * @param v Values
     */
    public Vector(double[] v)
    {
        this.v = v;
    }
    
    /**
     * Fills the vector with the specified value.
     * 
     * @param val The value to fill the vector with
     */
    public void fill(double val)
    {
        for (int a = 0; a < v.length; a++)
        {
            v[a] = val;
        }
    }
    
    /**
     * Returns a value in the vector.
     * 
     * @param i Position
     * @return The value
     */
    public double get(int i)
    {
        return v[i];
    }
    
    /**
     * Sets a value in the vector.
     * 
     * @param i Position
     * @param value The value
     */
    public void set(int i, double value)
    {
        v[i] = value;
    }
    
    /**
     * Adds a value to a position in the vector.
     * 
     * @param i Position
     * @param value The value to add
     */
    public void add(int i, double value)
    {
        v[i] += value;
    }
    
    /**
     * Returns the length of the vector.
     * 
     * @return Length
     */
    public int size()
    {
        return v.length;
    }
    
    /**
     * Calculates the sum of all values in this vector.
     * 
     * @return The sum
     */
    public double sum()
    {
        double sum = 0;
        for (int i = 0; i < size(); i++)
        {
            sum += v[i];
        }
        return sum;
    }
    
    /**
     * Calculates the dot product between this vector and another vector.
     * 
     * @param v2 The other vector
     * @return Dot product
     * @throws ArithmeticException If unable to calculate the dot product
     */
    public double dot(Vector v2) throws ArithmeticException
    {
        if (size() != v2.size())
        {
            throw new ArithmeticException("Size of both vectors must be the same");
        }
        
        double dot = 0;
        for (int i = 0; i < size(); i++)
        {
            dot += v[i] * v2.v[i];
        }
        return dot;
    }
    
    /**
     * Multiplies a row vector with a column vector.
     * 
     * @param v1 Row vector
     * @param v2 Column vector
     * @return Result matrix
     * @throws ArithmeticException If unable to calculate the product
     */
    public static Matrix mul(Vector v1, Vector v2) throws ArithmeticException
    {
        if (v1.size() != v2.size())
        {
            throw new ArithmeticException("Size of both vectors must be the same");
        }
        
        double[][] nv = new double[v1.size()][v1.size()];
        for (int r = 0; r < v1.size(); r++)
        {
            for (int c = 0; c < v1.size(); c++)
            {
                nv[r][c] = v1.v[r] * v2.v[c];
            }
        }
        return new Matrix(nv);
    }
    
    /**
     * Returns the index of the highest value in the vector.
     * 
     * @return Index of highest value
     */
    public int argmax()
    {
        double mV = v[0];
        int mI = 0;
        
        for (int i = 1; i < size(); i++)
        {
            if (v[i] > mV)
            {
                mV = v[i];
                mI = i;
            }
        }
        
        return mI;
    }
    
    /**
     * Updates the weights, assuming this is a bias vector.
     * 
     * @param dB Gradients vector
     * @param learningrate Learning rate
     */
    public void update_weights(Vector dB, double learningrate)
    {
        for (int i = 0; i < size(); i++)
        {
            v[i] -= dB.get(i) * learningrate;
        }
    }
    
    /**
     * Divides all values in the vector by a constant.
     * 
     * @param cons The constant
     */
    public void div(double cons)
    {
        for (int i = 0; i < size(); i++)
        {
            v[i] /= cons;
        }
    }
    
    /**
     * Multiplies all values in the vector by a constant.
     * 
     * @param cons The constant
     */
    public void mul(double cons)
    {
        for (int i = 0; i < size(); i++)
        {
            v[i] *= cons;
        }
    }
    
    /**
     * Calculates the L1 distance between two vectors.
     * 
     * @param v1 First vector
     * @param v2 Second vector
     * @return Squared L1 dist
     */
    public static double L1_dist(Vector v1, Vector v2) throws ArithmeticException
    {
        if (v1.size() != v2.size())
        {
            throw new ArithmeticException("Size of vectors must be equal");
        }
        
        double d = 0;
        for (int i = 0; i < v1.size(); i++)
        {
            d += Math.abs(v1.get(i) - v2.get(i));
        }
        
        return d;
    }
    
    /**
     * Calculates the (squared) L2 distance between two vectors.
     * 
     * @param v1 First vector
     * @param v2 Second vector
     * @return Squared L2 dist
     */
    public static double L2_dist(Vector v1, Vector v2) throws ArithmeticException
    {
        if (v1.size() != v2.size())
        {
            throw new ArithmeticException("Size of vectors must be equal");
        }
        
        double d = 0;
        for (int i = 0; i < v1.size(); i++)
        {
            d += Math.pow(v1.get(i) - v2.get(i), 2);
        }
        
        return d;
    }
    
    /**
     * Calculates the Frobenius norm (square root of the squared sums) for this vector.
     * 
     * @return The Frobenious norm
     */
    public double frobenius_norm()
    {
        double s = 0;
        for (int i = 0; i < v.length; i++)
        {
            s += Math.pow(v[i], 2);
        }
        
        return Math.sqrt(s);
    }
    
    /**
     * Normalizes a vector.
     * 
     * @param v The vector
     * @return Normalized vector
     */
    public static Vector normalize(Vector v)
    {
        double sqSum = 0;
        for (int i = 0; i < v.size(); i++)
        {
            sqSum += Math.pow(v.v[i], 2);
        }
        sqSum = Math.sqrt(sqSum);
        
        //Normalize values
        double[] sv = new double[v.size()];
        for (int i = 0; i < v.size(); i++)
        {
            sv[i] = v.v[i] / sqSum;
        }
        
        //Return normalized vector
        return new Vector(sv);
    }
    
    /**
     * Creates a copy of this vector.
     * 
     * @return The copy of this vector
     */
    public Vector copy()
    {
        double[] nv = new double[size()];
        System.arraycopy(v, 0, nv, 0, size());
        return new Vector(nv);
    }
    
    @Override
    public String toString()
    {
        String str = size() + " vector\n";
        str += "[";
        for (int i = 0; i < size(); i++)
        {
            str += df.format(v[i]);
            if (i < size() - 1)
            {
                str += ", ";
            }
        }
        str += "]";
        
        return str;
    }
}
