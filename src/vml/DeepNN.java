
package vml;

import java.text.DecimalFormat;

/**
 * Deep Neural Network Softmax classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class DeepNN extends Classifier
{
    //Training dataset
    private Matrix X;
    //Class values vector
    private Vector y;
    //Number of training iterations
    private int iterations = 100;
    //Hidden layer 1
    private HiddenLayer h1;
    //Hidden layer 2
    private HiddenLayer h2;
    //Output layer
    private OutLayer out;
    
    //For output
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Creates a new deep neural network.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param h1_size Size of first hidden layer
     * @param h2_size Size of second hidden layer
     * @param iterations Training iterations
     * @param learningrate Learning rate
     */
    public DeepNN(Dataset data, Dataset test, int h1_size, int h2_size, int iterations, double learningrate) 
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
        h1 = new HiddenLayer(noInputs, h1_size, learningrate);
        h2 = new HiddenLayer(h1_size, h2_size, learningrate);
        out = new OutLayer(h2_size, noCategories, learningrate);
        
        //Training iterations
        this.iterations = iterations;
        
        System.out.println("Deep Neural Network classifier");
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
        h1.forward(X);
        h2.forward(h1.scores);
        out.forward(h2.scores);
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @return Current loss
     */
    public double backward()
    {        
        double loss = out.backward(y);
        loss += h2.backward(out.w, out.dscores);
        loss += h1.backward(h2.w, h2.dhidden);
        
        out.updateWeights();
        h2.updateWeights();
        h1.updateWeights();
        
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
        Matrix bH1W = null;
        Matrix bH2W = null;
        Vector bOB = null;
        Vector bH1B = null;
        Vector bH2B = null;
        
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
                bH1W = h1.w.copy();
                bH2W = h2.w.copy();
                bH1B = h1.b.copy();
                bH2B = h2.b.copy();
            }
            
            //Output result
            if (i % out_step == 0 || i == iterations || i == 1) System.out.println("  iteration " + i + ": loss " + df.format(loss));
        }
        
        //Set best weights
        h1.w = bH1W;
        h1.b = bH1B;
        h2.w = bH2W;
        h2.b = bH2B;
        out.w = bOW;
        out.b = bOB;
        
        loss = iterate();
        System.out.println("  Best loss " + df.format(loss) + " at iteration " + best_iteration);
    }
}
