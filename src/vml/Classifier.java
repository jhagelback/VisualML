
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
    private DecimalFormat df = new DecimalFormat("0.00"); 
    //Randomizer
    public static Random rnd = new Random(2);
    //Training dataset
    protected Dataset data;
    //Test dataset
    protected Dataset test;
    
    /**
     * Trains the classifier on a dataset.
     */
    public abstract void train();
        
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
     * @return Accuracy
     */
    public double evaluate()
    {
        System.out.println("\nTraining dataset");
        long st = System.currentTimeMillis();
        double acc = evaluate(data);
        long el = System.currentTimeMillis() - st;
        System.out.println("Evaluation time: " + el + " ms");
        
        if (test != null)
        {
            System.out.println("\nTest dataset");
            st = System.currentTimeMillis();
            acc = evaluate(test);
            el = System.currentTimeMillis() - st;
            System.out.println("Evaluation time: " + el + " ms");
        }
        
        return acc;
    }
    
    /**
     * Evaluates the accuracy on a dataset.
     * 
     * @param d The dataset
     * @return Accuracy on the dataset
     */
    private double evaluate(Dataset d)
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
        
        System.out.println("Accuracy: " + correct + "/" + d.size() + "  " + df.format(perc) + "%");
        return perc;
    }
    
    /**
     * Returns the step size of outputs (since we don't want every iteration loss to
     * be printed to console if we have many iterations).
     * 
     * @param iterations Number of iterations
     * @return Output step size
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
