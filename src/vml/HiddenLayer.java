
package vml;

import cern.colt.matrix.*;

/**
 * Hidden layer using ReLU.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class HiddenLayer
{
    //Weights matrix
    public DoubleMatrix2D w;
    //Bias vector
    public DoubleMatrix1D b;
    //Gradients for gradient descent optimization
    private DoubleMatrix2D dW;
    private DoubleMatrix1D dB;
    //Training dataset
    private DoubleMatrix2D X;
    //Class values vector
    private DoubleMatrix1D y;
    //Scores matrix = X*W
    public DoubleMatrix2D scores;
    //ReLU gradients matrix
    public DoubleMatrix2D dhidden;
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
        w = op.matrix_rnd(noOutputs, noInputs, 0.1);
        //Init bias vector with 0's
        b = op.vector_zeros(noOutputs);
        
        //Learning rate
        this.learningrate = learningrate;
    }
    
    /**
     * Performs the forward pass (activation).
     * 
     * @param X Input data matrix
     */
    public void forward(DoubleMatrix2D X)
    {
        this.X = X;
        
        //Activation
        scores = op.mul(w, X);
        scores = op.add(scores, b);
        //ReLU activation
        op.max(scores, 0);
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @param w2 Weights for next layer
     * @param dscores Gradients for next layer
     * @return Current regularization loss
     */
    public double backward(DoubleMatrix2D w2, DoubleMatrix2D dscores)
    {
        //Evaluate gradients
        grad_relu(w2, dscores);
        //debug();
        //Update weights
        updateWeights();
        
        return 0.5 * RW;
    }
    
    /**
     * For debugging output.
     */
    private void debug()
    {
        System.out.println("\nWeights:");
        System.out.println(w);
        System.out.println(b);
        
        System.out.println("\nGradients: ");
        System.out.println(dW);
        System.out.println(dB);
        
        java.util.Scanner s = new java.util.Scanner(System.in);
        String i = s.next();
        if (i.equals("q")) System.exit(0);
    }
    
    /**
     * Calculates gradients.
     * 
     * @param w2 Weights for next layer
     * @param dscores Gradients for next layer
     */
    public void grad_relu(DoubleMatrix2D w2, DoubleMatrix2D dscores)
    {
        //Re-calculate regularization
        calc_regularization();
        
        //Backprop into hidden layer
        dhidden = op.mul(op.transpose(w2), dscores);
        //Backprop the ReLU non-linearity (set dhidden to 0 if activation is 0
        for (int r = 0; r < dhidden.rows(); r++)
        {
            for (int c = 0; c < dhidden.columns(); c++)
            {
                //Check if activation is <= 0
                if (scores.get(r, c) <= 0)
                {
                    //Switch off
                    dhidden.set(r, c, 0);
                }
            }
        }
        
        //Momentum
        DoubleMatrix2D oldDW = dW;
        
        //And finally the gradients
        dW = op.mul(dhidden, op.transpose(X));
        dB = op.sum_rows(dhidden);
        
        if (oldDW != null)
        {
            op.add(dW, oldDW, 0.1);
        }
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        op.add(dW, w, lambda * 0.5);
    }
    
    /**
     * Updates the weights matrix.
     */
    public void updateWeights()
    {
        //Update weights
        for (int r = 0; r < w.rows(); r++)
        {
            for (int c = 0; c < w.columns(); c++)
            {
                double old = w.get(r, c);
                w.set(r, c, old - dW.get(r, c) * learningrate);
            }
        }
        //Update bias
        for (int c = 0; c < b.size(); c++)
        {
            double old = b.get(c);
            b.set(c, old - dB.get(c) * learningrate);
        }
    }
    
    /**
     * Re-calcualtes the L2 regularization loss
     */
    private void calc_regularization()
    {
        //Regularization
        RW = 0;
        
        for (int r = 0; r < w.rows(); r++)
        {
            for (int c = 0; c < w.columns(); c++)
            {
                RW += Math.pow(w.get(r, c), 2);
            }
        }
        RW *= lambda;
    }
}
