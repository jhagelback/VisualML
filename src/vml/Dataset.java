
package vml;

import cern.colt.matrix.*;
import java.util.*;

/**
 * Container for datasets.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class Dataset 
{
    //List of instances
    ArrayList<Instance> data;
    //Min value for each attribute (used for normalizing attributes)
    private double[] min;
    //Max value for each attribute (used for normalizing attributes)
    private double[] max;
    //Used for checking number of categories
    private HashMap<Integer,Integer> cats;
    
    public static int Norm_NONE = 0;
    public static int Norm_POS = 1;
    public static int Norm_NEGPOS = 2;
    
    /**
     * Creates a new, empty dataset.
     */
    public Dataset()
    {
        data = new ArrayList<>();
        cats = new HashMap<>();
    }
    
    /**
     * Returns the number of possible categories (labels) for this dataset.
     * 
     * @return Number of possible categories
     */
    public int noCategories()
    {
        return cats.size();
    }
    
    /**
     * Returns the number of input attributes for the dataset.
     * 
     * @return Nunber of input attributes
     */
    public int noInputs()
    {
        //We need some data to do this...
        if (data.size() > 0)
        {
            return data.get(0).x.size();
        }
        return 0;
    }
    
    /**
     * Adds an instance to the dataset.
     * 
     * @param inst The instance to add
     */
    public void add(Instance inst)
    {
        data.add(inst);
        
        //Keep track of attribute min and max values (for normalizing)
        if (min == null)
        {
            //Init min to high values
            min = new double[inst.x.size()];
            for (int i = 0; i < min.length; i++)
            {
                min[i] = Double.MAX_VALUE;
            }
        }
        if (max == null)
        {
            //Init max to low values
            max = new double[inst.x.size()];
            for (int i = 0; i < max.length; i++)
            {
                max[i] = Double.MIN_VALUE;
            }
        }
        
        //Add values for this instance
        for (int c = 0; c < inst.x.size(); c++)
        {
            double v = inst.x.get(c);
            if (v < min[c]) min[c] = v;
            if (v > max[c]) max[c] = v;
        }
        
        //Add possible category values
        if (!cats.containsKey(inst.label))
        {
            cats.put(inst.label, 1);
        }
    }
    
    /**
     * Returns the instance at index i.
     * 
     * @param i The index
     * @return Instance at the index
     */
    public Instance get(int i)
    {
        return data.get(i);
    }
    
    /**
     * Returns the size of the dataset.
     * 
     * @return Size of dataset
     */
    public int size()
    {
        return data.size();
    }
    
    /**
     * Normalizes all attributes to a value between 0 and 1.
     * 
     * @param type Type of normalization (None, Pos, NegPos)
     */
    public void normalizeAttributes(int type)
    {
        if (type == Norm_NONE) return;
        else if (type == Norm_POS) normalizePos();
        else if (type == Norm_NEGPOS) normalizeNegPos();
    }
    
    /**
     * Normalizes all attributes to a value between 0 and 1.
     */
    public void normalizePos()
    {
        //Iterate over all training instances
        for (Instance inst : data)
        {
            //Create new scaled attributed vector
            DoubleMatrix1D nv = op.vector_zeros(inst.x.size());
            for (int c = 0; c < nv.size(); c++)
            {
                nv.set(c, (inst.x.get(c) - min[c]) / (max[c] - min[c])); // 0 ... 1
            }
            //Set new instance vector
            inst.x = nv;
        }
    }
    
    /**
     * Normalizes all attributes to a value between -1 and 1.
     */
    public void normalizeNegPos()
    {
        //Iterate over all training instances
        for (Instance inst : data)
        {
            //Create new scaled attributed vector
            DoubleMatrix1D nv = op.vector_zeros(inst.x.size());
            for (int c = 0; c < nv.size(); c++)
            {
                nv.set(c, (inst.x.get(c) - min[c]) / (max[c] - min[c]) * 2.0 - 1.0); // -1 ... 1
            }
            //Set new instance vector
            inst.x = nv;
        }
    }
    
    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder();
        for (Instance inst : data)
        {
            b.append(inst.toString());
            b.append("\n");
        }
        return b.toString();
    }
    
    /**
     * Creates an input matrix for this dataset.
     * 
     * @return Input matrix
     */
    public DoubleMatrix2D input_matrix()
    {
        //Create instances matrix
        DoubleMatrix2D X = op.matrix_zeros(noInputs(), size());
        for (int r = 0; r < size(); r++)
        {
            Instance inst = data.get(r);
            for (int c = 0; c < inst.x.size(); c++)
            {
                X.set(c, r, inst.x.get(c));
            }
        }
        
        return X;
    }
    
    /**
     * Creates a label vector for this dataset.
     * 
     * @return Label vector
     */
    public DoubleMatrix1D label_vector()
    {
        //Create label (correct class) vector
        DoubleMatrix1D y = op.vector_zeros(size());
        for (int r = 0; r < size(); r++)
        {
            Instance inst = data.get(r);
            double c_val = inst.label;
            y.set(r, c_val);
        }
        
        return y;
    }
}
