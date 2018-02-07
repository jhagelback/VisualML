
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
    double learningrate = 1.0;
    
    /**
     * Sets if regularization shall be used or not.
     */
    boolean use_regularization = true;
    
    /**
     * L2 regularization strength.
     */
    double lambda = 0.01;
    
    /**
     * Number of training iterations.
     */
    int iterations = 200;
    
    /**
     * Normalization type to use for the data.
     */
    int normalization_type = Dataset.Norm_NONE;
    
    /**
     * Size of batches for batch training.
     */
    int batch_size = 0;
    
    /**
     * Creates default settings.
     */
    public LSettings()
    {
        learningrate = 1.0;
        use_regularization = true;
        lambda = 0.01;
        normalization_type = Dataset.Norm_NONE;
        iterations = 200;
        batch_size = 0;
    }
}
