
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
     * Sets if data shall be normalized.
     */
    boolean use_normalization = false;
    
    /**
     * Sets lower and upper bounds for normalized values.
     */
    int[] normalization_bounds = new int[2];
    
    /**
     * Size of batches for batch training.
     */
    int batch_size = 0;
    
    /**
     * Creates default settings.
     */
    public NNSettings()
    {
        learningrate = 0.3;
        use_regularization = true;
        lambda = 0.001;
        use_momentum = true;
        use_normalization = false;
        normalization_bounds = new int[2];
        layers = new int[]{16};
        iterations = 1000;
        batch_size = 0;
    }
}
