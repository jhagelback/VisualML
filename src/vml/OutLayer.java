
package vml;

import java.util.Random;

/**
 * Output layer using Softmax.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class OutLayer
{
    //Weights tensor
    protected Tensor2D w;
    //Bias tensor
    protected Tensor1D b;
    //Gradients for gradient descent optimization
    private Tensor2D dW;
    private Tensor1D dB;
    //Training dataset
    private Tensor2D X;
    //Class values tensor
    private Tensor1D y;
    //Scores tensor = X*W+b
    protected Tensor2D scores;
    //Softmax gradients tensor
    protected Tensor2D dscores;
    //L2 regularization
    private double RW;
    //Configuration settings
    private NNSettings settings;
    
    /**
     * Constructor.
     * 
     * @param noInputs Number of input values
     * @param noCategories Number of categories
     * @param settings Configuration settings for this classifier
     * @param rnd Randomiser
     */
    public OutLayer(int noInputs, int noCategories, NNSettings settings, Random rnd) 
    {
        if (rnd != null)
        {
            //Init weight tensor
            w = Tensor2D.randomNormal(noCategories, noInputs, rnd);
            //Init bias tensor to 0's
            b = Tensor1D.zeros(noCategories);
        }
        
        //Settings
        this.settings = settings;
    }
    
    /**
     * Creates a copy of this layer.
     * 
     * @return Layer
     */
    public OutLayer copy()
    {
        OutLayer no = new OutLayer(1, 1, settings, null);
        no.w = w.copy();
        no.b = b.copy();
        return no;
    }
    
    /**
     * Performs the forward pass (activation).
     * 
     * @param X Input data tensor
     */
    public void forward(Tensor2D X)
    {
        this.X = X;
        
        //Activation
        scores = Tensor2D.activation(w, X, b);        
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @param y Labels tensor
     * @return Current loss
     */
    public double backward(Tensor1D y)
    {
        //Input data
        this.y = y;
        
        //Calculate loss and evaluate gradients
        double loss = grad_softmax();
        
        return loss;
    }
    
    /**
     * Classifies an instance in the dataset.
     * 
     * @param i Index of the instance
     * @return Predicted class value
     */
    public int classify(int i)
    {
        int pred_class = scores.argmax(i);
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
        
        //Calculate exponentials
        //Tensor2D logprobs = scores.exp();
        
        //To avoid numerical instability
        Tensor2D logprobs = scores.shift_columns();
        logprobs.exp();
        
        //Normalize
        logprobs.normalize();
        
        //Calculate cross-entropy loss tensor
        Tensor1D loss_vec = logprobs.calc_loss(y);
        
        //Average loss
        loss = loss_vec.sum() / num_train;
        //Regularization loss
        loss += RW;
        
        //Momentum
        Tensor2D oldDW = null;
        Tensor1D oldDB = null;
        if (dW != null && settings.momentum > 0.0)
        {
            oldDW = dW.copy();
            oldDB = dB.copy();
        }
        
        //Gradients
        dscores = logprobs.calc_dscores(y);
        dW = Tensor2D.mul_transpose(dscores, X);
        dB = dscores.sum_rows();
        
        //Momentum
        if (oldDW != null && settings.momentum > 0.0)
        {
            dW.add(oldDW, settings.momentum);
            dB.add(oldDB, settings.momentum);
        }
        
        //Add regularization to gradients
        //The weight tensor scaled by Lambda*0.5 is added
        if (settings.lambda > 0)
        {
            dW.add(w, settings.lambda * 0.5);
        }
        
        return loss;
    }
    
    /**
     * Calculates data loss using Softmax.
     * 
     * @return Current data loss
     */
    public double calc_loss()
    {
        //Init some variables
        int num_train = X.columns();
        double loss = 0;
        
        //Calculate exponentials
        //Tensor2D logprobs = scores.exp();
        
        //To avoid numerical instability
        Tensor2D logprobs = scores.shift_columns();
        logprobs.exp();
        
        //Normalize
        logprobs.normalize();
        
        //Calculate cross-entropy loss tensor
        Tensor1D loss_vec = logprobs.calc_loss(y);
        
        //Average loss
        loss = loss_vec.sum() / num_train;
        
        return loss;
    }
    
    /**
     * Updates the weights and bias tensors.
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
