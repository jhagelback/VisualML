
package vml;

/**
 * Hidden layer using ReLU.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
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
    //Class values vector
    private Vector y;
    //Scores matrix = X*W
    public Matrix scores;
    //ReLU gradients matrix
    public Matrix dhidden;
    //L2 regularization
    private double RW;
    //L2 regularization strength
    private double lambda = 0.001;
    //Learningrate
    private double learningrate = 0.1;
    
    /**
     * Constructor.
     * 
     * @param noInputs Number of input values
     * @param noOutputs Number of outut values
     * @param learningrate Learning rate
     */
    public HiddenLayer(int noInputs, int noOutputs, double learningrate) 
    {
        //Init weight matrix
        w = Matrix.random(noOutputs, noInputs, 0.1, Classifier.rnd);
        //Init bias vector to 0's
        b = Vector.zeros(noOutputs);
        
        //Learning rate
        this.learningrate = learningrate;
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
        Matrix oldDW = dW;
        
        //And finally the gradients
        dW = Matrix.mul_transpose(dhidden, X);
        dB = dhidden.sum_rows();
        
        if (oldDW != null)
        {
            dW.add(oldDW, 0.1);
        }
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        dW.add(w, lambda * 0.5);
    }
    
    /**
     * Updates the weights matrix.
     */
    public void updateWeights()
    {
        //Update weights
        w.update_weights(dW, learningrate);
        //Update bias
        b.update_weights(dB, learningrate);
    }
    
    /**
     * Re-calcualtes the L2 regularization loss
     */
    private void calc_regularization()
    {
        //Regularization
        RW = w.L2_norm() * lambda;
    }
}
