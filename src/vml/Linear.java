
package vml;

import java.text.DecimalFormat;

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
        //Set dataset
        this.data = data;
        this.test = test;
        
        //Size of dataset
        int noCategories = data.noCategories();
        int noInputs = data.noInputs();
        
        //Init weight matrix
        w = Matrix.random(noCategories, noInputs, 0.1, Classifier.rnd);
        //Init bias vector to 0's
        b = Vector.zeros(noCategories);
        
        //Settings
        this.settings = settings;
        batch_size = settings.batch_size;
    }
    
    /**
     * Initializes the weight matrix and bias vector as the example in
     * http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/
     */
    public Linear()
    {
        double[][] w_init = {
                    {1.00,  2.00},
                    {2.00, -4.00},
                    {3.00, -1.00}
        };
        
        double[] b_init = {0.00, 0.50, -0.50};
        
        //Init weight matrix
        w = new Matrix(w_init);
        //Init bias vector
        b = new Vector(b_init);
        //Settings
        settings = new LSettings();
        settings.learningrate = 1.0;
        settings.iterations = 20;
        
        //Read data
        data = ClassifierFactory.readDataset("data/demo.csv", new DataSource());
    }
    
    /**
     * Trains the classifier.
     * 
     * @param o Logger for log info
     */
    @Override
    public void train(Logger o)
    {
        o.appendText("Linear Softmax classifier");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        o.appendText("\nTraining classifier");
        
        //For output
        int out_step = settings.iterations / 10;
        
        //Optimization Gradient Descent
        double best_loss = Double.MAX_VALUE;
        int best_iteration = 0;
        double loss = 0;
        Matrix bestW = null;
        Vector bestB = null;
        
        for (int i = 1; i <= settings.iterations; i++)
        {
            loss = iterate();
            
            //Check if we have a new best loss
            if (loss < best_loss)
            {
                best_loss = loss;
                best_iteration = i;
                //Copy best weights
                bestW = w.copy();
                bestB = b.copy();
            }
            
            //Output result
            if (i % out_step == 0 || i == settings.iterations || i == 1) o.appendText("    iteration " + i + ":  loss " + df.format(loss));
        }
        
        //Set best weights
        //Don't use in batch training since loss heavily depends on which batch is trained on
        if (batch_size == 0)
        {
            w = bestW;
            b = bestB;
            activation();
            o.appendText("  Best loss " + df.format(best_loss) + " at iteration " + best_iteration);
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
        if (batch_size > 0)
        {
            Dataset b = getNextBatch();
            X = b.input_matrix();
            y = b.label_vector();
        }
        else
        {
            X = data.input_matrix();
            y = data.label_vector();
        }

        //Forward pass (activation)
        activation();
        //Calculate loss and evaluate gradients
        //double loss = grad_svm();
        double loss = grad_softmax();
        //Update weights
        updateWeights();
        
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
        
        //Gradients
        Matrix dscores = logprobs.calc_dscores(y);
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
        dW.divide(num_train);
        dB.divide(num_train);
        
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
        
        if (settings.use_regularization)
        {
            RW = w.L2_norm() * settings.lambda;
        }
    }
}
