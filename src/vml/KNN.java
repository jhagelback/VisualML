
package vml;

import java.util.ArrayList;
import java.util.Collections;

/**
 * k-Nearest Neighbor classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class KNN extends Classifier
{
    /**
     * Internal class to hold training examples and distances.
     */
    private class KInstance implements Comparable
    {
        //Attributes
        Vector x;
        //Label
        int label;
        //Distance
        double dist;
        
        /**
         * Creates a new instance.
         * 
         * @param x Attributes
         * @param label Label
         */
        public KInstance(Vector x, int label)
        {
            this.x = x;
            this.label = label;
        }
        
        @Override
        public int compareTo(Object o)
        {
            KInstance ki = (KInstance)o;
            Double a = dist;
            Double b = ki.dist;
            return a.compareTo(b);
        }
    }
    
    //Training data
    private ArrayList<KInstance> d;
    //Test data
    private Dataset test;
    //K-value
    private int K = 3;
    //Number of categories
    private int noCategories;
    
    /**
     * Creates a classifier.
     * 
     * @param K K-value
     * @param noCategories Number of categories
     */
    public KNN(int K, int noCategories)
    {
        this.K = K;
        this.noCategories = noCategories;
    }
    
    /**
     * Sets training dataset.
     * 
     * @param data Training dataset
     */
    @Override
    public void setData(Dataset data)
    {
        //Init internal data array
        d = new ArrayList<>(data.size());
        //Iterate over all training instances
        for (Instance inst : data.data)
        {
            //Create internal instance
            KInstance ki = new KInstance(inst.x, inst.label);
            d.add(ki);
        }
    }
    
    /**
     * Trains the classifier.
     */
    @Override
    public void train()
    {
        //Nothing is done here
    }
    
    /**
     * Executes one training iteration.
     * 
     * @return Current loss
     */
    @Override
    public double iterate()
    {
        //Nothing is done here
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
        this.test = test;
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
        Instance inst = test.get(i);
        
        //Iterate over the training data
        //and calculate distances
        d.stream().parallel().forEach((ki) -> {
            ki.dist = Vector.L2_dist(inst.x, ki.x);
        });
        
        //Sort list based on distance
        Collections.sort(d);
        
        //Create result array with number of
        //occurences for each label, plus distances
        Vector res = Vector.zeros(noCategories);
        Vector dist = Vector.zeros(noCategories);
        for (int j = 0; j < K; j++)
        {
            int pred_y = d.get(j).label;
            res.set(pred_y, res.get(pred_y) + 1);
            dist.set(pred_y, dist.get(pred_y) + d.get(j).dist);
        }
        
        double bestNo = 0;
        double bestD = Double.MAX_VALUE;
        int bestY = -1;
        
        for (int j = 0; j < noCategories; j++)
        {
            double no = res.get(j);
            double d = dist.get(j);
            //More occurences
            if (no > bestNo)
            {
                bestY = j;
                bestD = d;
                bestNo = no;
            }
            //Same occurences, check distance
            if (no == bestNo)
            {
                if (d < bestD)
                {
                    bestY = j;
                    bestD = d;
                    bestNo = no;
                }
            }
        }
        
        //Return predicted label
        return bestY;
    }
}
