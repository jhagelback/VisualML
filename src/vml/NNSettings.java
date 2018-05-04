
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
    public double learningrate = 0.3;
    
    /**
     * Loss threshold for stopping training.
     */
    public double stop_threshold = 0.000005;
    
    /**
     * Sets if regularization shall be used or not.
     */
    public boolean use_regularization = false;
    
    /**
     * L2 regularization strength.
     */
    public double lambda = 0.001;
    
    /**
     * Sets if momentum shall be used by the hidden layers.
     */
    public boolean use_momentum = true;
    
    /**
     * Number and size of the hidden layers.
     */
    public int[] layers = {16};
    
    /**
     * Number of training epochs.
     */
    public int epochs = 1000;
    
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
     * Sets if training dataset shall be shuffled or not.
     */
    public boolean shuffle = true;
    
    /**
     * Creates default settings.
     */
    public NNSettings()
    {
        learningrate = 0.3;
        stop_threshold = 0.000005;
        use_regularization = true;
        lambda = 0.001;
        use_momentum = true;
        use_normalization = false;
        normalization_bounds = new int[2];
        layers = new int[]{16};
        epochs = 1000;
        batch_size = 0;
        shuffle = true;
    }
}
