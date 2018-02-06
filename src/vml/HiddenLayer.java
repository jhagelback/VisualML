
package vml;

/**
 * Hidden layer using ReLU.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class HiddenLayer
{
    //Weights matrix
    public Matrix w;
    //Bias vector
    public Vector b;
    //Gradients for gradient descent optimization
    private Matrix dW;
    private Vector dB;
    //Training dataset
    private Matrix X;
    //Scores matrix = X*W
    public Matrix scores;
    //ReLU gradients matrix
    public Matrix dhidden;
    //L2 regularization
    private double RW;
    //Configuration settings
    private NNSettings settings;
    
    /**
     * Constructor.
     * 
     * @param noInputs Number of input values
     * @param noOutputs Number of outut values
     * @param settings Configuration settings for this classifier
     */
    public HiddenLayer(int noInputs, int noOutputs, NNSettings settings) 
    {
        //Init weight matrix
        w = Matrix.random(noOutputs, noInputs, 0.1, Classifier.rnd);
        //Init bias vector to 0's
        b = Vector.zeros(noOutputs);
        
        //Settings
        this.settings = settings;
    }
    
    public HiddenLayer copy()
    {
        HiddenLayer nh = new HiddenLayer(1, 1, settings);
        nh.w = w.copy();
        nh.b = b.copy();
        return nh;
    }
    
    /**
     * Performs the forward pass (activation).
     * 
     * @param X Input data matrix
     */
    public void forward(Matrix X)
    {
        this.X = X;
        
        //Activation
        scores = Matrix.activation(w, X, b);
        
        //ReLU activation
        scores.max(0);
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @param w2 Weights for next layer
     * @param dscores Gradients for next layer
     * @return Current regularization loss
     */
    public double backward(Matrix w2, Matrix dscores)
    {
        //Evaluate gradients
        grad_relu(w2, dscores);
        
        return 0.5 * RW;
    }
    
    /**
     * Calculates gradients.
     * 
     * @param w2 Weights for next layer
     * @param dscores Gradients for next layer
     */
    public void grad_relu(Matrix w2, Matrix dscores)
    {
        //Re-calculate regularization
        calc_regularization();
        
        //Backprop into hidden layer
        dhidden = Matrix.transpose_mul(w2, dscores);
        //Backprop the ReLU non-linearity (set dhidden to 0 if activation is 0
        dhidden.backprop_relu(scores);
        
        //Momentum
        Matrix oldDW = null;
        if (dW != null && settings.use_momentum)
        {
            oldDW = dW.copy();
        }
        
        //And finally the gradients
        dW = Matrix.mul_transpose(dhidden, X);
        dB = dhidden.sum_rows();
        
        if (oldDW != null && settings.use_momentum)
        {
            dW.add(oldDW, 0.1);
        }
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        if (settings.use_regularization)
        {
            dW.add(w, settings.lambda * 0.5);
        }
    }
    
    /**
     * Updates the weights matrix.
     */
    public void updateWeights()
    {
        //Update weights
        w.update_weights(dW, settings.learningrate);
        //Update bias
        b.update_weights(dB, settings.learningrate);
    }
    
    /**
     * Re-calcualtes the L2 regularization loss
     */
    private void calc_regularization()
    {
        //Regularization
        RW = 0;
        
        if (settings.use_regularization)
        {
            RW = w.L2_norm() * settings.lambda;
        }
    }
}
