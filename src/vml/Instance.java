
package vml;

import cern.colt.matrix.*;

/**
 * Hold a data instance (example).
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class Instance 
{
    //Attributes vector
    protected DoubleMatrix1D x;
    //Label (class value)
    protected int label;
    
    /**
     * Creates a new instance.
     * 
     * @param attr Attribute values
     * @param label Label (class value)
     */
    public Instance(double[] attr, int label)
    {
        x = op.vector(attr);
        this.label = label;
    }
    
    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder();
        b.append("[");
        for (int i = 0; i < x.size() - 1; i++)
        {
            b.append(x.get(i));
            b.append(", ");
        }
        b.append(x.get(x.size() - 1));
        b.append(" -> ");
        b.append(label);
        b.append("]");
        
        return b.toString();
    }
}
