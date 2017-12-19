
package vml;

import cern.colt.matrix.*;
import java.text.DecimalFormat;

/**
 * Linear classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class Linear extends Classifier
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
    private DoubleMatrix2D scores;
    //L2 regularization
    private double RW;
    //L2 regularization strength
    private double lambda = 0.01;
    //Learningrate
    private double learningrate = 0.1;
    //Delta for SVM loss calculations
    private double delta = 1.0;
    //Set to true to use L2 regularization
    private boolean use_regularization;
    //Number of training iterations
    private int iterations = 20;
    
    //For output
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Constructor.
     * 
     * @param noInputs Number of input values
     * @param noCategories Number of categories
     * @param iterations Number of training iterations
     * @param learningrate Learning rate
     */
    public Linear(int noInputs, int noCategories, int iterations, double learningrate) 
    {
        //Init weight matrix
        w = op.matrix_rnd(noCategories, noInputs, 0.1);
        //Init bias vector to 0's
        b = op.vector_zeros(noCategories);
        
        //Learning rate
        this.learningrate = learningrate;
        //Training iterations
        this.iterations = iterations;
    }
    
    /**
     * Initializes the weight matrix as the example in
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
        w = op.matrix(w_init);
        //Init bias vector
        b = op.vector(b_init);
        //Learning rate
        learningrate = 1.0;
    }
    
    /**
     * Sets training dataset.
     * 
     * @param data Training dataset
     */
    @Override
    public void setData(Dataset data)
    {
        X = data.input_matrix();
        y = data.label_vector();
    }
    
    /**
     * Trains the classifier.
     */
    @Override
    public void train()
    {
        //For output
        int out_step = getOutputStep(iterations);
        
        use_regularization = true;
        
        //Optimization Gradient Descent
        double best_loss = Double.MAX_VALUE;
        int best_iteration = 0;
        double loss = 0;
        DoubleMatrix2D bestW = null;
        DoubleMatrix1D bestB = null;
        
        for (int i = 1; i <= iterations; i++)
        {
            loss = iterate();
            
            //Error check
            if (Double.isNaN(loss) || Double.isInfinite(loss))
            {
                //Weights have exploded. Stop training
                System.err.println("Warning: weights overflow. Training is stopped.");
                break;
            }
            
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
            if (i % out_step == 0 || i == iterations || i == 1) System.out.println("  iteration " + i + ":  loss " + df.format(loss));
        }
        
        //Set best weights
        w = bestW;
        b = bestB;
        
        loss = iterate();
        System.out.println("  Best loss " + df.format(loss) + " at iteration " + best_iteration);
    }
    
    /**
     * Executes one training iteration.
     * 
     * @return Current loss
     */
    @Override
    public double iterate()
    {
        //Forward pass (activation)
        activation();
        //Calculate loss and evaluate gradients
        //double loss = grad_svm();
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
     * Performs activation for the specified dataset.
     * 
     * @param test Test dataset
     */
    @Override
    public void activation(Dataset test)
    {
        //Activation
        scores = op.mul(w, test.input_matrix());
        scores = op.add(scores, b);
    }
    
    /**
     * Performs forward pass (activation).
     */
    private void activation()
    {
        //Activation
        scores = op.mul(w, X);
        scores = op.add(scores, b);  
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
        loss += RW;
        
        //Gradients
        DoubleMatrix2D dscores = probs;
        for (int j = 0; j < num_train; j++)
        {
            int corr_index = (int)y.get(j);
            dscores.set(corr_index, j, dscores.get(corr_index, j) - 1);
        }
        op.divide(dscores, num_train);
        
        dW = op.mul(dscores, op.transpose(X));
        dB = op.sum_rows(dscores);
        
        //Add regularization to gradients
        //The weight matrix scaled by Lambda*0.5 is added
        op.add(dW, w, lambda * 0.5);
        
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
        dW = op.matrix_zeros(w.rows(), w.columns());
        dB = op.vector_zeros(w.rows());
        
        //Init some variables
        int num_classes = w.rows();
        int num_train = X.columns();
        double loss = 0;
        
        //Iterate over all training examples
        for (int i = 0; i < num_train; i++)
        {
            //Get vectors
            DoubleMatrix1D score = op.column(scores, i);
            DoubleMatrix1D xi = op.column(X, i);
             
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
                        op.add(dW, xi, j, 1.0);
                        op.add(dW, xi, corr_index, -1.0);
                        //Update bias vector
                        op.add(dB, j, 1);
                        op.add(dB, corr_index, -1);
                    }
                }
            }
        }
        
        //Average gradients
        op.divide(dW, num_train);
        op.divide(dB, num_train);
        
        //Average loss + reqularization
        loss = loss / num_train + RW;
        
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
        
        if (use_regularization)
        {
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
}
