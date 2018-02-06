
package vml;

/**
 * Output layer using Softmax.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class OutLayer
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
    //Scores matrix = X*W+b
    public Matrix scores;
    //Softmax gradients matrix
    public Matrix dscores;
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
     */
    public OutLayer(int noInputs, int noCategories, NNSettings settings) 
    {
        //Init weight matrix
        w = Matrix.random(noCategories, noInputs, 0.1, Classifier.rnd);
        //Init bias vector to 0's
        b = Vector.zeros(noCategories);
        
        //Settings
        this.settings = settings;
    }
    
    public OutLayer copy()
    {
        OutLayer no = new OutLayer(1, 1, settings);
        no.w = w.copy();
        no.b = b.copy();
        return no;
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
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @param y Labels vector
     * @return Current loss
     */
    public double backward(Vector y)
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
        //Matrix logprobs = scores.exp();
        
        //To avoid numerical instability
        Matrix logprobs = scores.shift_columns();
        logprobs.exp();
        
        //Normalize
        logprobs.normalize();
        
        //Calculate cross-entropy loss vector
        Vector loss_vec = logprobs.calc_loss(y);
        
        //Average loss
        loss = loss_vec.sum() / num_train;
        //Regularization loss
        loss += RW;
        
        //Gradients
        dscores = logprobs.calc_dscores(y);
        dW = Matrix.mul_transpose(dscores, X);
        dB = dscores.sum_rows();
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        if (settings.use_regularization)
        {
            dW.add(w, settings.lambda * 0.5);
        }
        
        return loss;
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
