
package vml;

import cern.colt.matrix.*;

/**
 * Output layer using Softmax.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class OutLayer
{
    //Weights matrix
    public DoubleMatrix2D w;
    //Bias vector
    public DoubleMatrix1D b;
    //Gradients for gradient descent optimization
    private DoubleMatrix2D dW;
    private DoubleMatrix1D dB;
    //Input data
    private DoubleMatrix2D X;
    //Class values vector
    private DoubleMatrix1D y;
    //Scores matrix
    public DoubleMatrix2D scores;
    //Softmax gradients matrix
    public DoubleMatrix2D dscores;
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
     * @param noCategories Number of categories
     * @param learningrate Learning rate
     */
    public OutLayer(int noInputs, int noCategories, double learningrate) 
    {
        //Init weight matrix
        w = op.matrix_rnd(noCategories, noInputs, 0.1);
        //Init bias vector with 0's
        b = op.vector_zeros(noCategories);
        
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
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @param y Labels vector
     * @return Current loss
     */
    public double backward(DoubleMatrix1D y)
    {
        //Input data
        this.y = y;
        
        //Calculate loss and evaluate gradients
        double loss = grad_softmax();
        //debug(loss);
        //Update weights
        updateWeights();
        
        return loss;
    }
    
    /**
     * For debugging output.
     * 
     * @param loss Current loss
     */
    private void debug(double loss)
    {
        System.out.println("Loss: " + loss);
        
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
     * Classifies an instance in the dataset.
     * 
     * @param i Index of the instance
     * @return Predicted class value
     */
    public int classify(int i)
    {
        DoubleMatrix1D act = op.column(scores, i);
        int pred_class = op.argmax(act);
        return pred_class;    
    }
    
    /**
     * Evaluates loss and calculates gradients using Softmax.
     * 
     * @return Current loss
     */
    private double grad_softmax()
    {
        //Re-calculate regularization
        calc_regularization();
        
        //Init some variables
        int num_train = X.columns();
        double loss = 0;
        
        //To avoid numerical instability
        DoubleMatrix2D exp_scores = op.shift_columns(scores);
        exp_scores = op.exp(scores);
        
        //Calculate exponentials
        //DoubleMatrix2D exp_scores = op.exp(scores);
        //Normalize
        DoubleMatrix2D probs = op.average_columns(exp_scores);
        
        DoubleMatrix1D corect_logprobs = op.vector_zeros(num_train);
        for (int j = 0; j < num_train; j++)
        {
            DoubleMatrix1D prob = probs.viewColumn(j);
            int corr_index = (int)y.get(j);
            double class_score = prob.get(corr_index);
            double Li = -1.0 * Math.log(class_score) / Math.log(Math.E);
            corect_logprobs.set(j, Li);
        }
        //Average loss
        loss = op.sum(corect_logprobs) / num_train;
        //Regularization loss
        loss += 0.5 * RW;
        
        //Gradients
        dscores = probs;
        for (int j = 0; j < num_train; j++)
        {
            int corr_index = (int)y.get(j);
            dscores.set(corr_index, j, dscores.get(corr_index, j) - 1);
        }
        op.divide(dscores, num_train);
        
        //Momentum
        DoubleMatrix2D oldDW = dW;
        
        dW = op.mul(dscores, op.transpose(X));
        dB = op.sum_rows(dscores);
        
        if (oldDW != null)
        {
            op.add(dW, oldDW, 0.1);
        }
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        op.add(dW, w, lambda * 0.5);
        
        return loss;
    }
    
    /**
     * Updates the weights matrix.
     */
    private void updateWeights()
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
