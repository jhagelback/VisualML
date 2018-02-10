
package vml;

/**
 * Contains settings for the Linear Softmax classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class LSettings 
{
    /**
     * Learning rate.
     */
    public double learningrate = 1.0;
    
    /**
     * Sets if regularization shall be used or not.
     */
    public boolean use_regularization = true;
    
    /**
     * L2 regularization strength.
     */
    public double lambda = 0.01;
    
    /**
     * Number of training iterations.
     */
    public int iterations = 200;
    
    /**
     * Sets if data shall be normalized.
     */
    public boolean use_normalization = false;
    
    /**
     * Sets lower and upper bounds for normalized values.
     */
    public int[] normalization_bounds = new int[2];
    
    /**
     * Size of batches for batch training.
     */
    public int batch_size = 0;
    
    /**
     * Creates default settings.
     */
    public LSettings()
    {
        learningrate = 1.0;
        use_regularization = true;
        lambda = 0.01;
        use_normalization = false;
        normalization_bounds = new int[2];
        iterations = 200;
        batch_size = 0;
    }
}
