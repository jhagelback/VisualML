
package vml;

import java.text.DecimalFormat;
import java.util.Random;

/**
 * Base class for classifiers.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public abstract class Classifier 
{
    //Output formatting
    private static DecimalFormat df = new DecimalFormat("0.00"); 
    //Randomizer
    public static Random rnd = new Random(2);
    //Training dataset
    protected Dataset data;
    //Test dataset
    protected Dataset test;
    //Batch size for batch training. Set to 0 to disable batch training.
    protected int batch_size = 0;
    //Current batch number
    protected int batch_no = 0;
    
    /**
     * Get next batch for batch training
     * 
     * @return Next batch of training instances
     */
    public Dataset getNextBatch()
    {
        Dataset b = new Dataset();
        
        //Start and end indexes
        int start = batch_no * batch_size;
        int end = (batch_no + 1) * batch_size;
        for (int i = start; i < end; i++)
        {
            if (i < data.size())
            {
                b.add(data.get(i));
            }
        }
        batch_no++;
        if (end >= data.size()) batch_no = 0;
        
        return b;
    }
    
    /**
     * Formats elapes time to a suitable string.
     * 
     * @param ms Elapsed time in milliseconds
     * @return Time string
     */
    public static String time_string(long ms)
    {
        String s = ms + " ms";
        if (ms >= 1000 && ms < 60000)
        {
            double sec = (double)ms / 1000.0;
            s = df.format(sec) + " sec";
        }
        if (ms >= 60000)
        {
            int mins = (int)(ms / 60000);
            int secs = (int)(ms % 60000);
            double sec = (double)secs / 1000.0;
            s = mins + " min " + df.format(sec) + " sec";
        }
        return s;
    }
    
    /**
     * Trains the classifier on a dataset.
     * 
     * @param out Logger for log info
     */
    public abstract void train(Logger out);
        
    /**
     * Returns the training dataset.
     * 
     * @return Training dataset
     */
    public Dataset getData()
    {
        return data;
    }
    
    /**
     * Classifies an instance in the dataset.
     * 
     * @param i Index of the instance
     * @return Predicted class value
     */
    public abstract int classify(int i);
    
    /**
     * Classifies an instance.
     * 
     * @param i The instance
     * @return Predicted category label
     */
    public String classify(Instance i)
    {
        Dataset t = new Dataset();
        t.add(i);
        
        activation(t);
        
        int pred_label = classify(0);
        
        return data.getCategoryLabel(pred_label);
    }
    
    /**
     * Executes one training iteration.
     * 
     * @return Current loss
     */
    public abstract double iterate();
    
    /**
     * Performs activation (forward pass) for the specified test dataset.
     * 
     * @param test Test dataset
     */
    public abstract void activation(Dataset test);
    
    /**
     * Evaluates the accuracy on the training dataset and, if specified, test dataset.
     * 
     * @param eval_train Sets if accuracy shall be evaluated on the training dataset
     * @param eval_test Sets if accuracy shall be evaluated on the test dataset
     * @param out Logger for log info
     */
    public void evaluate(boolean eval_train, boolean eval_test, Logger out)
    {
        long st,el;
        
        if (eval_train)
        {
            out.appendText("\nTraining dataset");
            st = System.currentTimeMillis();
            evaluate(data, out);
            el = System.currentTimeMillis() - st;
            out.appendText("Evaluation time: " + time_string(el));
        }
        
        if (test != null && eval_test)
        {
            out.appendText("\nTest dataset");
            st = System.currentTimeMillis();
            evaluate(test, out);
            el = System.currentTimeMillis() - st;
            out.appendText("Evaluation time: " + time_string(el));
        }
    }
    
    /**
     * Evaluates the accuracy on a dataset.
     * 
     * @param d The dataset
     * @param out Logger for log info
     * @return Accuracy on the dataset
     */
    private double evaluate(Dataset d, Logger out)
    {
        //Activation for the dataset
        activation(d);
        int correct = 0;
        
        for (int i = 0; i < d.size(); i++)
        {
            int pred_class = classify(i);
            if (pred_class == d.get(i).label)
            {
                correct++;
            }
        }
        double perc = (double)correct / (double)d.size() * 100.0;
        
        out.appendText("Accuracy: " + correct + "/" + d.size() + "  " + df.format(perc) + "%");
        return perc;
    }
    
    /**
     * Calculates the accuracy on the training dataset.
     * 
     * @return Accuracy (in %)
     */
    public double train_accuracy()
    {
        //Activation for the dataset
        activation(data);
        int correct = 0;
        
        for (int i = 0; i < data.size(); i++)
        {
            int pred_class = classify(i);
            if (pred_class == data.get(i).label)
            {
                correct++;
            }
        }
        double perc = (double)correct / (double)data.size() * 100.0;
        
        return perc;
    }
    
    /**
     * Calculates the accuracy on the test dataset.
     * 
     * @return Accuracy (in %)
     */
    public double test_accuracy()
    {
        //Error check
        if (test == null) return 0;
        
        //Activation for the dataset
        activation(test);
        int correct = 0;
        
        for (int i = 0; i < test.size(); i++)
        {
            int pred_class = classify(i);
            if (pred_class == test.get(i).label)
            {
                correct++;
            }
        }
        double perc = (double)correct / (double)test.size() * 100.0;
        
        return perc;
    }
    
    /**
     * Returns the step size of outputs (since we don't want every iteration loss to
     * be printed to console if we have many iterations).
     * 
     * @param iterations Number of iterations
     * @return Logger step size
     */
    public int getOutputStep(int iterations)
    {
        if (iterations <= 25) return 1;
        else if (iterations <= 50) return 2;
        else if (iterations <= 100) return 5;
        else if (iterations <= 100) return 10;
        else if (iterations <= 400) return 20;
        else if (iterations <= 1000) return 50;
        else if (iterations <= 5000) return 100;
        else if (iterations <= 10000) return 200;
        else return 500;
    }
}
