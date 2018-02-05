
package vml;

/**
 * Contains settings for the Neural Network classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class NNSettings 
{
    /**
     * Learning rate.
     */
    double learningrate = 0.3;
    
    /**
     * Setts if regularization shall be used or not.
     */
    boolean use_regularization = false;
    
    /**
     * L2 regularization strength for the hidden layers.
     */
    double hidden_lambda = 0.001;
    
    /**
     * L2 regularization strength for the output layer.
     */
    double output_lambda = 0.001;
    
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
        hidden_lambda = 0.001;
        output_lambda = 0.001;
        use_momentum = true;
        normalization_type = Dataset.Norm_NONE;
        layers = new int[]{16};
        iterations = 1000;
    }
}
