
package vml;

import java.util.ArrayList;

/**
 * RBF (Radial-Basis Function) Kernel classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Kernel extends Classifier
{
    //Configuration settings
    private KernelSettings settings;
    //Internal test dataset
    private Dataset tdata;
    //Kernels
    private ArrayList<RBF> kernels;
    
    /**
     * Creates a classifier.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param settings Configuration settings for this classifier
     */
    public Kernel(Dataset data, Dataset test, KernelSettings settings)
    {
        //Set dataset
        this.data = data;
        this.test = test;
        
        //Settings
        this.settings = settings;
    }
    
    /**
     * Trains the classifier.
     * 
     * @param o Logger for log info
     */
    @Override
    public void train(Logger o)
    {
        o.appendText("RBF Kernel Classifier");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        
        //Reset kernels
        kernels = null;
        iterate();
    }
    
    /**
     * Executes one training iteration.
     * 
     * @return Current loss
     */
    @Override
    public double iterate()
    {
        if (kernels == null)
        {
            //Init new sets of kernels
            kernels = new ArrayList<>();

            //Create one kernel for each combination of possible categories
            for (int c0 = 0; c0 < data.noCategories(); c0++)
            {
                for (int c1 = 0; c1 < data.noCategories(); c1++)
                {
                    if (c0 != c1 && c0 < c1)
                    {
                        //Create kernel
                        kernels.add(new RBF(c0, c1, data, settings.gamma));
                    }
                }
            }
        }
        
        return 0;
    }
    
    /**
     * Performs activation for the specified dataset.
     * 
     * @param test Test dataset
     */
    @Override
    public void activation(Dataset test)
    {
        //Sets test dataset
        tdata = test;
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
        //Get instance
        Instance inst = tdata.get(i);
        
        //Votes for the Max-vote strategy to choose between categories
        Vector votes = Vector.zeros(data.noCategories());
        
        //Iterate over all kernels and classify the instance
        for (int k = 0; k < kernels.size(); k++)
        {
            int pred = kernels.get(k).classify(inst);
            votes.v[pred]++;
        }
        
        //Predicted category is the category with highest number of votes
        int y_pred = votes.argmax();
        
        //Return prediction
        return y_pred;
    }
}
