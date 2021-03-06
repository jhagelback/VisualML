
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
    private Tensor2D X;
    //Class values tensor
    private Tensor1D y;
    //Hidden layers
    private HiddenLayer[] hidden;
    //Output layer
    private OutLayer out;
    //Configuration settings
    private NNSettings settings;
    //Current iteration
    private int current_iter = 1;
    
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
        //Iterable training phase
        iterable = true;
        
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
        Tensor2D train = X;
        
        X = test.input_tensor();
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
        double loss = 0;
        
        if (batch_size > 0)
        {
            int no_batches = data.size() / batch_size;
            if (data.size() % batch_size != 0) no_batches++;
            
            //Train each batch
            for (int i = 0; i < no_batches; i++)
            {
                Dataset batch = getNextBatch();
                X = batch.input_tensor();
                y = batch.label_tensor();
                
                forward();
                
                //Dropout
                if (settings.dropout > 0.0)
                {
                    for (HiddenLayer h : hidden)
                    {
                        h.dropout();
                    }
                }
                
                backward();
            }
            
            //Calculate loss
            forward();
            //We only need to take loss from output layer into consideration, since
            //loss on hidden layers are purely based on regularization
            loss = out.backward(y);
        }
        else
        {
            //Train whole dataset
            X = data.input_tensor();
            y = data.label_tensor();
            
            forward();
            
            //Dropout
            if (settings.dropout > 0.0)
            {
                for (HiddenLayer h : hidden)
                {
                    h.dropout();
                }
            }
            
            backward();
            
            //Calculate loss
            forward();
            //We only need to take loss from output layer into consideration, since
            //loss on hidden layers are purely based on regularization
            loss = out.backward(y);
        }
        
        //Learning rate decay
        if (settings.learningrate_decay > 0)
        {
            settings.learningrate -= settings.learningrate_decay;
            if (settings.learningrate < 0.0) settings.learningrate = 0.0;
        }
        
        current_iter++;
        if (current_iter >= settings.epochs)
        {
            training_done = true;
        }
        
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
        training_done = false;
        current_iter = 1;
        
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
        int out_step = settings.epochs / 10;
        if (out_step <= 0) out_step = 1;
        
        //Optimization Gradient Descent
        double loss = 0;
        double p_loss = Double.MAX_VALUE;
        
        o.appendText("  Iteration  Loss");
        
        while (!training_done)
        {
            //Training iteration
            loss = iterate();
            
            //Output result
            if (current_iter % out_step == 0 || current_iter == settings.epochs || current_iter == 1) 
            {
                String str = "  " + Logger.format_spaces(current_iter + ":", 9);
                str += "  " + df.format(loss);
                o.appendText(str);
            }
            
            //Check stopping criterion
            if (current_iter > 2)
            {
                double diff = loss - p_loss;
                if (diff <= settings.stop_threshold && diff >= 0)
                {
                    //i = settings.epochs;
                    training_done = true;
                    o.appendText("  Stop threshold reached at iteration " + current_iter);
                    break;
                }
                p_loss = loss;
            }
        }
    }
}
