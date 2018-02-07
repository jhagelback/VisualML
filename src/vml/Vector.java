
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
        double[] v = new double[s];
        for (int a = 0; a < s; a++)
        {
            v[a] = rnd.nextDouble() * scale;
        }
        return new Vector(v);
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
    public void divide(double cons)
    {
        for (int i = 0; i < size(); i++)
        {
            v[i] /= cons;
        }
    }
    
    /**
     * Calculates the L1 distance between two vectors.
     * 
     * @param v1 First vector
     * @param v2 Second vector
     * @return Squared L1 dist
     */
    public static double L1_dist(Vector v1, Vector v2)
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
    public static double L2_dist(Vector v1, Vector v2)
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
