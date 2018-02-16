
package vml;

import java.util.*;

/**
 * Container for datasets.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Dataset 
{
    //List of instances
    public ArrayList<Instance> data;
    //Min value for each attribute (used for normalizing attributes)
    private double[] min;
    //Max value for each attribute (used for normalizing attributes)
    private double[] max;
    //Used for checking number of categories
    private HashMap<Integer,Integer> cats;
    //Mapping from int label to string label
    private HashMap<Integer,String> intToCat;
    //Private string name
    private String name;
    
    /**
     * Creates a new, empty dataset.
     */
    public Dataset()
    {
        data = new ArrayList<>();
        cats = new HashMap<>();
        this.name = "";
    }
    
    /**
     * Creates a new, empty dataset.
     * 
     * @param name Name of dataset
     */
    public Dataset(String name)
    {
        data = new ArrayList<>();
        cats = new HashMap<>();
        this.name = name;
    }
    
    /**
     * Get the filename for this dataset.
     * 
     * @return Filename
     */
    public String getName()
    {
        return name;
    }
    
    /**
     * Sets the mapping between integer labels and category labels.
     * 
     * @param intToCat The mapping
     */
    public void setLabelMapping(HashMap<Integer,String> intToCat)
    {
        this.intToCat = intToCat;
    }
    
    /**
     * Returns the category label for an integer label.
     * 
     * @param label Integer label
     * @return Category label, or null of not found
     */
    public String getCategoryLabel(int label)
    {
        return intToCat.get(label);
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
     * Creates a clone of this dataset without any instances added. All settings
     * are copied to the clone.
     * 
     * @return Empty clone of this dataset
     */
    public Dataset clone_empty()
    {
        Dataset d = new Dataset();
        d.cats = cats;
        d.intToCat = intToCat;
        d.max = max;
        d.min = min;
        d.name = name;
        
        return d;
    }
    
    /**
     * Returns a subset containing all instances within the specified range.
     * 
     * @param start Start index
     * @param end End index
     * @return Subset of this dataset
     */
    public Dataset getSubset(int start, int end)
    {
        Dataset sub = clone_empty();
        for (int i = start; i < end; i++)
        {
            sub.data.add(data.get(i));
        }
        
        return sub;
    }
    
    /**
     * Returns a subset containing all instances not within the specified range.
     * 
     * @param start Start index
     * @param end End index
     * @return Subset of this dataset
     */
    public Dataset getInverseSubset(int start, int end)
    {
        Dataset sub = clone_empty();
        for (int i = 0; i < data.size(); i++)
        {
            if (i < start || i >= end)
            {
                sub.data.add(data.get(i));
            }
        }
        
        return sub;
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
     * Normalizes all attributes.
     * 
     * @param min_value Lower bound for normalized values
     * @param max_value Upper bound for normalized values
     */
    public void normalizeAttributes(int min_value, int max_value)
    {
        //Calculate range and shift
        int range = Math.abs(max_value - min_value);
        int shift = min_value;
        
        //Iterate over all training instances
        for (Instance inst : data)
        {
            //Create new scaled attributed vector
            double[] nv = new double[inst.x.size()];
            for (int i = 0; i < nv.length; i++)
            {
                nv[i] = (inst.x.get(i) - min[i]) / (max[i] - min[i]); // 0 ... 1
                nv[i] *= range;
                nv[i] += shift;
            }
            //Set new instance vector
            inst.x = new Vector(nv);
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
    public Matrix input_matrix()
    {
        //Create instances matrix
        Matrix X = Matrix.zeros(noInputs(), size());
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
    public Vector label_vector()
    {
        //Create label (correct class) vector
        Vector y = Vector.zeros(size());
        for (int r = 0; r < size(); r++)
        {
            Instance inst = data.get(r);
            double c_val = inst.label;
            y.set(r, c_val);
        }
        
        return y;
    }
}
