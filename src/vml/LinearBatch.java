
package vml;

import cern.colt.matrix.*;
import java.text.DecimalFormat;
import java.util.ArrayList;

/**
 * Linear classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class LinearBatch extends Classifier
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
    
    private int current_batch;
    private int batch_size;
    private int batches;
    private ArrayList<Batch> batch;
    
    private class Batch
    {
        DoubleMatrix2D x;
        DoubleMatrix1D y;
        
        public Batch(DoubleMatrix2D x, DoubleMatrix1D y)
        {
            this.x = x;
            this.y = y;
        }
    }
    
    /**
     * Constructor.
     * 
     * @param noInputs Number of input values
     * @param noCategories Number of categories
     * @param iterations Number of training iterations
     * @param learningrate Learning rate
     */
    public LinearBatch(int noInputs, int noCategories, int iterations, double learningrate) 
    {
        //Init weight matrix
        w = op.matrix_rnd(noCategories, noInputs, 0.1);
        //Init bias vector to 0's
        b = op.vector_zeros(noCategories);
        
        //Learning rate
        this.learningrate = learningrate;
        //Training iterations
        this.iterations = iterations;
        
        current_batch = 0;
        batch_size = 100;
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
        
        batches = data.size() / batch_size;
        if (batches % batch_size > 0) batches++;
        
        batch = new ArrayList<>(batches);
        
        for (int b = 0; b < batches; b++)
        {
            Dataset s = new Dataset();
            for (int i = b * batch_size; i < (b+1)*batch_size; i++)
            {
                if (b < data.size())
                {
                    s.add(data.get(b));
                }
            }
            batch.add(new Batch(s.input_matrix(), s.label_vector()));
        }
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
        for (int b = 0; b < batches; b++)
        {
            //Forward pass (activation)
            activation(batch.get(b).x);
            //Calculate loss and evaluate gradients
            grad_softmax(batch.get(b).x, batch.get(b).y);
            //Update weights
            updateWeights();
        }
        
        //Forward pass (activation)
        activation(X);
        //Calculate loss
        double loss = grad_softmax(X, y);
        
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
        scores = op.mul(w, test.input_matrix());
        scores = op.add(scores, b);
    }
    
    /**
     * Performs forward pass (activation).
     */
    private void activation(DoubleMatrix2D cX)
    {
        //Activation
        scores = op.mul(w, cX);
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
    private double grad_softmax(DoubleMatrix2D cX, DoubleMatrix1D cY)
    {
        //Re-calculate regularization
        calc_regularization();
        
        //Init some variables
        int num_train = cX.columns();
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
            int corr_index = (int)cY.get(j);
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
            int corr_index = (int)cY.get(j);
            dscores.set(corr_index, j, dscores.get(corr_index, j) - 1);
        }
        op.divide(dscores, num_train);
        
        dW = op.mul(dscores, op.transpose(cX));
        dB = op.sum_rows(dscores);
        
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
