
package vml;

import java.text.DecimalFormat;

/**
 * Neural Network Softmax classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class NN extends Classifier
{
    //Training dataset
    private Matrix X;
    //Class values vector
    private Vector y;
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
     * @param data Training dataset
     * @param test Test dataset
     * @param hidden_size Size of hidden layer
     * @param iterations Training iterations
     * @param learningrate Learning rate
     */
    public NN(Dataset data, Dataset test, int hidden_size, int iterations, double learningrate) 
    {
        //Set dataset
        this.data = data;
        this.test = test;
        X = data.input_matrix();
        y = data.label_vector();
        
        //Size of dataset
        int noCategories = data.noCategories();
        int noInputs = data.noInputs();
        
        //Create layers
        hidden = new HiddenLayer(noInputs, hidden_size, learningrate);
        out = new OutLayer(hidden_size, noCategories, learningrate);
        
        //Training iterations
        this.iterations = iterations;
        
        System.out.println("Neural Network classifier");
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
        Matrix train = X;
        
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
        
        out.updateWeights();
        hidden.updateWeights();
        
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
        int pred_class = out.scores.argmax(i);
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
        
        Matrix bOW = null;
        Matrix bHW = null;
        Vector bOB = null;
        Vector bHB = null;
        
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
