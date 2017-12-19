
package vml;

import java.text.DecimalFormat;

/**
 * Base class for classifiers.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public abstract class Classifier 
{
    //Output formatting
    private DecimalFormat df = new DecimalFormat("0.00"); 
    
    /**
     * Trains the classifier on a dataset.
     */
    public abstract void train();
    
    /**
     * Sets training dataset.
     * 
     * @param data Training dataset
     */
    public abstract void setData(Dataset data);
    
    /**
     * Classifies an instance in the dataset.
     * 
     * @param i Index of the instance
     * @return Predicted class value
     */
    public abstract int classify(int i);
    
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
     * Evaluates the accuracy on a test dataset.
     * 
     * @param test The test dataset
     * @return Accuracy on the test dataset
     */
    public double evaluate(Dataset test)
    {
        //Activation for the test dataset
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
        
        System.out.println("Accuracy: " + correct + "/" + test.size() + "  " + df.format(perc) + "%");
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
