
package vml;

/**
 * Contains settings for the RBF Kernel classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class KernelSettings 
{
    /**
     * Gamma value.
     */
    public double gamma = 1.0;
    
    /**
     * Sets if data shall be normalized.
     */
    public boolean use_normalization = false;
    
    /**
     * Sets lower and upper bounds for normalized values.
     */
    public int[] normalization_bounds = new int[2];
    
    /**
     * Creates default settings.
     */
    public KernelSettings()
    {
        gamma = 1.0;
        use_normalization = false;
        normalization_bounds = new int[2];
    }
}
