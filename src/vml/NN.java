
package vml;

import java.text.DecimalFormat;
import java.util.Random;

/**
 * Neural Network Softmax classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class NN extends Classifier
{
    //Training dataset
    private Matrix X;
    //Class values vector
    private Vector y;
    //Hidden layers
    private HiddenLayer[] hidden;
    //Output layer
    private OutLayer out;
    //Configuration settings
    private NNSettings settings;
    
    //For output
    private DecimalFormat df = new DecimalFormat("0.0000"); 
    
    /**
     * Creates a new neural network.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param settings Configuration settings for this classifier
     */
    public NN(Dataset data, Dataset test, NNSettings settings) 
    {
        //Set dataset
        this.data = data;
        this.test = test;
        
        //Size of dataset
        noCategories = data.noCategories();
        noInputs = data.noInputs();
        
        //Settings
        this.settings = settings;
        batch_size = settings.batch_size;
        
        //Initalises layers
        init();
    }
    
    /**
     * Initialises layers.
     */
    private void init()
    {
        Random rnd = new Random(seed);
        //Hidden layers
        hidden = new HiddenLayer[settings.layers.length];
        hidden[0] = new HiddenLayer(noInputs, settings.layers[0], settings, rnd);
        if (hidden.length > 1)
        {
            for (int i = 1; i < hidden.length; i++)
            {
                hidden[i] = new HiddenLayer(settings.layers[i-1], settings.layers[i], settings, rnd);
            }
        }
        //Out layer
        out = new OutLayer(settings.layers[settings.layers.length - 1], noCategories, settings, rnd);
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
        //First hidden layer (data as input)
        hidden[0].forward(X);
        //Subsequent hidden layers (previous hidden layer as input)
        for (int i = 1; i < hidden.length; i++)
        {
            hidden[i].forward(hidden[i-1].scores);
        }
        //Output layer (last hidden layer as input)
        out.forward(hidden[hidden.length-1].scores);
    }
    
    /**
     * Performs the backwards pass.
     * 
     * @return Current loss
     */
    public double backward()
    {
        //Output layer
        double loss = out.backward(y);
        //Last hidden layer (gradients from output layer)
        loss += hidden[hidden.length-1].backward(out.w, out.dscores);
        //Rest of the hidden layers (gradients from next layer)
        for (int i = hidden.length - 2; i >= 0; i--)
        {
            loss += hidden[i].backward(hidden[i+1].w, hidden[i+1].dhidden);
        }
        
        //Weights updates
        out.updateWeights();
        for (int i = hidden.length - 1; i >= 0; i--)
        {
            hidden[i].updateWeights();
        }
        
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
     * 
     * @param o Logger for log info
     */
    @Override
    public void train(Logger o)
    {
        //Initalises layers
        init();
        
        o.appendText("Neural Network classifier (" + hidden.length + " hidden layers)");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        o.appendText("\nTraining classifier");
        
        //For output
        int out_step = settings.iterations / 10;
        
        //Optimization Gradient Descent
        OutLayer bOut = null;
        HiddenLayer[] bHidden = new HiddenLayer[hidden.length];
        OutLayer cOut = null;
        HiddenLayer[] cHidden = new HiddenLayer[hidden.length];
        
        double loss = 0;
        double best_loss = Double.MAX_VALUE;
        int best_iteration = 0;
        
        for (int i = 1; i <= settings.iterations; i++)
        {
            //Copy current weights and biases
            cOut = out.copy();
            cHidden = new HiddenLayer[hidden.length];
            for (int h = 0; h < hidden.length; h++)
            {
                cHidden[h] = hidden[h].copy();
            }
            
            //Training iteration
            loss = iterate();
            
            //Check if we have a new best loss
            if (loss < best_loss)
            {
                best_loss = loss;
                best_iteration = i;
                //Set best layers
                bOut = cOut;
                bHidden = cHidden;
            }
            
            //Output result
            if (i % out_step == 0 || i == settings.iterations || i == 1) o.appendText("    iteration " + i + ":  loss " + df.format(loss));
        }
        
        //Set best weights
        //Don't use in batch training since loss heavily depends on which batch is trained on
        if (batch_size == 0)
        {
            out = bOut;
            hidden = bHidden;
            forward();
            o.appendText("  Best loss " + df.format(best_loss) + " at iteration " + best_iteration);
        }
    }
}
