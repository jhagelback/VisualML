
package vml;

import java.util.stream.IntStream;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * RBF (Radial-Basis Function) kernel for two categories in the dataset.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class RBF 
{
    //Instances of class 0
    private Dataset d0;
    //Label of class 0
    private int l0;
    //Instances of class 1
    private Dataset d1;
    //Label of class 1
    private int l1;
    //Offset value
    private double offset;
    //Gamma setting
    private double gamma = 1.0;
    
    /**
     * Creates a new RBF kernel.
     * 
     * @param l0 Label for category 0
     * @param l1 Label for category 1
     * @param data Training dataset
     * @param gamma Gamma setting
     */
    public RBF(int l0, int l1, Dataset data, double gamma)
    {
        //Gamma setting
        this.gamma = gamma;
        
        //Labels
        this.l0 = l0;
        this.l1 = l1;
        
        //Create one dataset for each category
        d0 = data.clone_empty();
        d1 = data.clone_empty();
        for (Instance inst : data.data)
        {
            if (inst.label == l0) d0.data.add(inst);
            if (inst.label == l1) d1.data.add(inst);
        }
        
        //Calculate offset
        calc_offset();
    }
    
    /**
     * Calculates the offset value for this RBF kernel.
     */
    private void calc_offset()
    {
        //Calculate sum of RBF values for class 0
        DoubleAdder s0 = new DoubleAdder();
        IntStream.range(0, d0.size()).parallel().forEach(i1 -> {
           for (int i2 = 0; i2 < d0.size(); i2++)
            {
                s0.add(RBF(d0.get(i1).x, d0.get(i2).x));
            } 
        });
        //Calculate sum of RBF values for class 1
        DoubleAdder s1 = new DoubleAdder();
        IntStream.range(0, d1.size()).parallel().forEach(i1 -> {
           for (int i2 = 0; i2 < d1.size(); i2++)
            {
                s1.add(RBF(d1.get(i1).x, d1.get(i2).x));
            } 
        });
        
        //Calculate offset
        offset = (1.0 / Math.pow(d1.size(), 2)) * s1.doubleValue() - (1.0 / Math.pow(d0.size(), 2)) * s0.doubleValue();
    }
    
    /**
     * Classifies this instance as either category l0 or l1.
     * 
     * @param i Instance to classify
     * @return Predicted category
     */
    public int classify(Instance i)
    {
        //Calculate RBF value
        double y = calc_RBF(i);
        //Check sign of RBF to predict category
        if (y > 0) return l0;
        else return l1;
    }
    
    /**
     * Calculates the RBF value for an instance.
     * 
     * @param i The instance
     * @return RBF value
     */
    private double calc_RBF(Instance i)
    {
        //Iterate over all training data instances
        //and calculate RBF values
        DoubleAdder s0 = new DoubleAdder();
        IntStream.range(0, d0.size()).parallel().forEach(d -> {
            s0.add(RBF(i.x, d0.data.get(d).x));
        });
        DoubleAdder s1 = new DoubleAdder();
        IntStream.range(0, d1.size()).parallel().forEach(d -> {
            s1.add(RBF(i.x, d1.data.get(d).x));
        });
        
        //Calculate RBF value
        double rbf = s0.doubleValue() / d0.size() - s1.doubleValue() / d1.size() + offset;
        
        return rbf;
    }
    
    /**
     * (Gaussian) Radial Basis Function
     * 
     * @param v1 Vector 1
     * @param v2 Vector 2
     * @return Calculated RB value
     */
    public double RBF(Vector v1, Vector v2)
    {
        double sq_dist = 0;
        //Find squared distance between v1 and v2
        //First, sum the squared diff between all values
        for (int i = 0; i < v1.size(); i++)
        {
            sq_dist += Math.pow(v1.v[i] - v2.v[i], 2);
        }
        double rb = Math.pow(Math.E, -gamma * sq_dist);
        
        return rb;
    }
}
