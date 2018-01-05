
package vml;

import cern.colt.matrix.*;
import java.text.DecimalFormat;

/**
 * Two-layer neural network classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class NN extends Classifier
{
    //Training dataset
    private DoubleMatrix2D X;
    //Class values vector
    private DoubleMatrix1D y;
    //Number of training iterations
    private int iterations = 100;
    //Hidden layer
    private HiddenLayer hidden;
    //Output layer
    private OutLayer out;
    
    //For output
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Creates a new neural network.
     * 
     * @param noInputs Number of input variables in the dataset
     * @param noCategories Number of categories (class values)
     * @param hidden_size Size of hidden layer
     * @param iterations Training iterations
     * @param learningrate Learning rate
     */
    public NN(int noInputs, int noCategories, int hidden_size, int iterations, double learningrate) 
    {
        this.iterations = iterations;
        
        //Create layers
        hidden = new HiddenLayer(noInputs, hidden_size, learningrate);
        out = new OutLayer(hidden_size, noCategories, learningrate);
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
     * Performs activation (forward pass) for the specified test dataset.
     * 
     * @param test Test dataset
     */
    @Override
    public void activation(Dataset test)
    {
        //To avoid errors when using the GUI, we need to keep a reference to the training data
        DoubleMatrix2D train = X;
        
        X = test.input_matrix();
        forward();
        
        //Put back the training data
        X = train;
    }
    
    /**
     * Performs the forward pass (activation).
     */
    public void forward()
    {        
        hidden.forward(X);
        out.forward(hidden.scores);
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @return Current loss
     */
    public double backward()
    {        
        double loss = out.backward(y);
        loss += hidden.backward(out.w, out.dscores);
        return loss;
    }
    
    /**
     * Performs one training iteration.
     * 
     * @return Current loss
     */
    @Override
    public double iterate()
    {
        forward();
        double loss = backward();
        return loss;
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
        DoubleMatrix1D act = out.scores.viewColumn(i);
        int pred_class = op.argmax(act);
        return pred_class;
    }
    
    /**
     * Trains the classifier.
     */
    @Override
    public void train()
    {
        //For output
        int out_step = getOutputStep(iterations);
        
        //Optimization Gradient Descent
        
        DoubleMatrix2D bOW = null;
        DoubleMatrix2D bHW = null;
        DoubleMatrix1D bOB = null;
        DoubleMatrix1D bHB = null;
        
        double loss = 0;
        double best_loss = Double.MAX_VALUE;
        int best_iteration = 0;
        
        for (int i = 1; i <= iterations; i++)
        {
            loss = iterate();
            
            //Error check
            if (Double.isNaN(loss))
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
                bOW = out.w.copy();
                bOB = out.b.copy();
                bHW = hidden.w.copy();
                bHB = hidden.b.copy();
            }
            
            //Output result
            if (i % out_step == 0 || i == iterations || i == 1) System.out.println("  iteration " + i + ":  loss " + df.format(loss));
        }
        
        //Set best weights
        hidden.w = bHW;
        hidden.b = bHB;
        out.w = bOW;
        out.b = bOB;
        
        loss = iterate();
        System.out.println("  Best loss " + df.format(loss) + " at iteration " + best_iteration);
    }
}
