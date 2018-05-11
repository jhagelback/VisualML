
package vml;

import java.util.Random;

/**
 * Hidden layer using ReLU.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class HiddenLayer
{
    //Weights matrix
    protected Matrix w;
    //Bias vector
    protected Vector b;
    //Gradients for gradient descent optimization
    private Matrix dW;
    private Vector dB;
    //Training dataset
    private Matrix X;
    //Scores matrix = X*W
    protected Matrix scores;
    //ReLU gradients matrix
    protected Matrix dhidden;
    //L2 regularization
    private double RW;
    //Configuration settings
    private NNSettings settings;
    //Randomizer
    private Random rnd;
    
    /**
     * Constructor.
     * 
     * @param noInputs Number of input values
     * @param noOutputs Number of outut values
     * @param settings Configuration settings for this classifier
     * @param rnd Randomiser
     */
    public HiddenLayer(int noInputs, int noOutputs, NNSettings settings, Random rnd) 
    {
        if (rnd != null)
        {
            //Init weight matrix
            w = Matrix.randomNormal(noOutputs, noInputs, rnd);
            //Init bias vector to 0's
            b = Vector.zeros(noOutputs);
            this.rnd = rnd;
        }
        
        //Settings
        this.settings = settings;
    }
    
    /**
     * Creates a copy of this layer.
     * 
     * @return Layer
     */
    public HiddenLayer copy()
    {
        HiddenLayer nh = new HiddenLayer(1, 1, settings, null);
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
     * Applies dropout to this layer. Dropout is a regularization where
     * we zero-out random units during the training phase.
     */
    public void dropout()
    {
        for (int r = 0; r < scores.rows(); r++)
        {
            if (rnd.nextDouble() < settings.dropout)
            {
                for (int c = 0; c < scores.columns(); c++)
                {
                    scores.v[r][c] = 0;
                }
            }
        }
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
        Vector oldDB = null;
        if (dW != null && settings.momentum > 0.0)
        {
            oldDW = dW.copy();
            oldDB = dB.copy();
        }
        
        //And finally the gradients
        dW = Matrix.mul_transpose(dhidden, X);
        dB = dhidden.sum_rows();
        
        //Momentum
        if (oldDW != null && settings.momentum > 0.0)
        {
            dW.add(oldDW, settings.momentum);
            dB.add(oldDB, settings.momentum);
        }
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        if (settings.lambda > 0)
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
        
        if (settings.lambda > 0)
        {
            RW = w.L2_norm() * settings.lambda;
        }
    }
}
