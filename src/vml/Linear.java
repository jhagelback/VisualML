
package vml;

import java.text.DecimalFormat;
import java.util.Random;

/**
 * Linear Softmax classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Linear extends Classifier
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
    //Class values vector
    private Vector y;
    //Scores matrix = X*W+b
    private Matrix scores;
    //L2 regularization
    private double RW;
    //Configuration settings
    private LSettings settings;
    //Delta for SVM loss calculations
    private double delta = 1.0;
    
    //For output
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Constructor.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param settings Configuration settings for this classifier
     */
    public Linear(Dataset data, Dataset test, LSettings settings) 
    {
        //Iterable training phase
        iterable = true;
        
        //Set dataset
        this.data = data;
        this.test = test;
        
        //Size of dataset
        noCategories = data.noCategories();
        noInputs = data.noInputs();
        
        //Initializes weights and biases
        init();
        
        //Settings
        this.settings = settings;
        batch_size = settings.batch_size;        
    }
    
    /**
     * Initializes weights and biases.
     */
    private void init()
    {
        Random rnd = new java.util.Random(seed);
        //Init weight matrix
        w = Matrix.randomNormal(noCategories, noInputs, rnd);
        //Init bias vector to 0's
        b = Vector.zeros(noCategories);
    }
    
    /**
     * Trains the classifier.
     * 
     * @param o Logger for log info
     */
    @Override
    public void train(Logger o)
    {
        //Initializes weights and biases
        init();
        
        o.appendText("Linear Softmax regression classifier");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        o.appendText("\nTraining classifier");
        
        //For output
        int out_step = settings.epochs / 10;
        if (out_step <= 0) out_step = 1;
        
        //Optimization Gradient Descent
        double loss = 0;
        double p_loss = Double.MAX_VALUE;
        
        o.appendText("  Iteration  Loss");
        
        for (int i = 1; i <= settings.epochs; i++)
        {
            //Training iteration
            loss = iterate();
            
            //Output result
            if (i % out_step == 0 || i == settings.epochs || i == 1) 
            {
                String str = "  " + Logger.format_spaces(i + ":", 9);
                str += "  " + df.format(loss);
                o.appendText(str);
            }
            
            //Check stopping criterion
            if (i > 2)
            {
                double diff = loss - p_loss;
                if (diff <= settings.stop_threshold && diff >= 0)
                {
                    //i = settings.epochs;
                    o.appendText("  Stop threshold reached at iteration " + i);
                    break;
                }
                p_loss = loss;
            }
        }
    }
    
    /**
     * Executes one training iteration.
     * 
     * @return Current loss
     */
    @Override
    public double iterate()
    {
        double loss = 0;
        
        if (batch_size > 0)
        {
            int no_batches = data.size() / batch_size;
            if (data.size() % batch_size != 0) no_batches++;
            
            //Train each batch
            for (int i = 0; i < no_batches; i++)
            {
                Dataset batch = getNextBatch();
                X = batch.input_matrix();
                y = batch.label_vector();
                
                //Forward pass (activation)
                activation();
                //Calculate loss and evaluate gradients
                //double loss = grad_svm();
                grad_softmax();
                
                //Update weights
                updateWeights();
            }
            
            //Calculate loss
            activation();
            //We only need to take loss from output layer into consideration, since
            //loss on hidden layers are purely based on regularization
            loss = calc_loss();
        }
        else
        {
            //Train whole dataset
            X = data.input_matrix();
            y = data.label_vector();
            
            //Forward pass (activation)
            activation();
            //Calculate loss and evaluate gradients
            //double loss = grad_svm();
            grad_softmax();
            
            //Update weights
            updateWeights();
            
            //Calculate loss
            activation();
            //We only need to take loss from output layer into consideration, since
            //loss on hidden layers are purely based on regularization
            loss = calc_loss();
        }
        
        //Learning rate decay
        if (settings.learningrate_decay > 0)
        {
            settings.learningrate -= settings.learningrate_decay;
            if (settings.learningrate < 0.0) settings.learningrate = 0.0;
        }
        
        return loss;
    }
        
    /**
     * Performs activation for the specified dataset.
     * 
     * @param test Test dataset
     */
    @Override
    public void activation(Dataset test)
    {
        //Activation
        scores = Matrix.activation(w, test.input_matrix(), b);
    }
    
    /**
     * Performs forward pass (activation).
     */
    private void activation()
    {
        //Activation
        scores = Matrix.activation(w, X, b);
    }
    
    /**
     * Classifies an instance in the dataset.
     * 
     * @param i Index of the instance
     * @return Predicted class value
     */
    @Override
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
        
        //Momentum
        Matrix oldDW = null;
        Vector oldDB = null;
        if (dW != null && settings.momentum > 0.0)
        {
            oldDW = dW.copy();
            oldDB = dB.copy();
        }
        
        //Gradients
        Matrix dscores = logprobs.calc_dscores(y);
        dW = Matrix.mul_transpose(dscores, X);
        dB = dscores.sum_rows();
        
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
        
        return loss;
    }
    
    /**
     * Evaluates loss and calculates gradients using SVM.
     * 
     * @return Current loss
     */
    private double grad_svm()
    {
        //Re-calculate regularization
        calc_regularization();
        
        //Gradients matrix
        dW = Matrix.zeros(w.rows(), w.columns());
        dB = Vector.zeros(w.rows());
        
        //Init some variables
        int num_classes = w.rows();
        int num_train = X.columns();
        double loss = 0;
        
        //Iterate over all training examples
        for (int i = 0; i < num_train; i++)
        {
            //Get vectors
            Vector score = scores.getColumn(i);
            Vector xi = X.getColumn(i);
             
            //Correct label (class value)
            int corr_index = (int)y.get(i);
            double correct_class_score = score.get(corr_index);
            
            //Gradients
            //Iterate over all class values
            for (int j = 0; j < num_classes; j++)
            {
                if (j != corr_index)
                {
                    //Calculate margin
                    double margin = score.get(j) - correct_class_score + delta;
                    if (margin > 0)
                    {
                        //Update gradients matrix
                        loss += margin;
                        dW.addToRow(xi, j, 1.0);
                        dW.addToRow(xi, corr_index, -1.0);
                        //Update bias vector
                        dB.add(j, 1);
                        dB.add(corr_index, -1);
                    }
                }
            }
        }
        
        //Average gradients
        dW.div(num_train);
        dB.div(num_train);
        
        //Average loss + reqularization
        loss = loss / num_train + RW;
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        dW.add(w, settings.lambda * 0.5);
        
        return loss;
    }
    
    /**
     * Updates the weights matrix.
     */
    private void updateWeights()
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
