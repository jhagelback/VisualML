
package vml;

/**
 * Contains settings for the Neural Network classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class NNSettings 
{
    /**
     * Learning rate.
     */
    double learningrate = 0.3;
    
    /**
     * Sets if regularization shall be used or not.
     */
    boolean use_regularization = false;
    
    /**
     * L2 regularization strength.
     */
    double lambda = 0.001;
    
    /**
     * Sets if momentum shall be used by the hidden layers.
     */
    boolean use_momentum = true;
    
    /**
     * Number and size of the hidden layers.
     */
    int[] layers = {16};
    
    /**
     * Number of training iterations.
     */
    int iterations = 1000;
    
    /**
     * Normalization type to use for the data.
     */
    int normalization_type = Dataset.Norm_NONE;
    
    /**
     * Creates default settings.
     */
    public NNSettings()
    {
        learningrate = 0.3;
        use_regularization = true;
        lambda = 0.001;
        use_momentum = true;
        normalization_type = Dataset.Norm_NONE;
        layers = new int[]{16};
        iterations = 1000;
    }
}
