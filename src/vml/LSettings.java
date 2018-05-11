
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
     * Loss threshold for stopping training.
     */
    public double stop_threshold = 0.000005;
    
    /**
     * L2 regularization strength.
     */
    public double lambda = 0.01;
    
    /**
     * Sets momentum rate (0.0 for no momentum).
     */
    public double momentum = 0.1;
    
    /**
     * Learning rate decay after each epoch (0.0 for no decay).
     */
    public double learningrate_decay = 0.0;
    
    /**
     * Number of training epochs.
     */
    public int epochs = 200;
    
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
    public LSettings()
    {
        learningrate = 1.0;
        stop_threshold = 0.000005;
        lambda = 0.01;
        momentum = 0.1;
        learningrate_decay = 0.0;
        use_normalization = false;
        normalization_bounds = new int[2];
        epochs = 200;
        batch_size = 0;
        shuffle = true;
    }
}
