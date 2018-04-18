
package vml;

import java.util.ArrayList;

/**
 * k-Nearest Neighbor classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
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
    //Internal test dataset
    private Dataset tdata;
    //Configuration settings
    private KNNSettings settings;
    
    /**
     * Creates a classifier.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param settings Configuration settings for this classifier
     */
    public KNN(Dataset data, Dataset test, KNNSettings settings)
    {
        //Set dataset
        this.data = data;
        this.test = test;
        
        //Size of dataset
        noCategories = data.noCategories();
        
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
        o.appendText("k-Nearest Neightbor Classifier");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        
        //Reset internal array
        d = null;
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
        //Init internal data array (if not already done)
        if (d == null)
        {
            d = new ArrayList<>(data.size());
            //Iterate over all training instances
            for (Instance inst : data.data)
            {
                //Create internal instance
                KInstance ki = new KInstance(inst.x, inst.label);
                d.add(ki);
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
        
        //Iterate over the training data
        //and calculate distances
        if (settings.distance_measure == KNNSettings.L1)
        {
            d.stream().parallel().forEach((ki) -> {
                ki.dist = Vector.L1_dist(inst.x, ki.x);
            });
        }
        if (settings.distance_measure == KNNSettings.L2)
        {
            d.stream().parallel().forEach((ki) -> {
                ki.dist = Vector.L2_dist(inst.x, ki.x);
            });
        }
        
        //Sort list based on distance
        d.sort((i1, i2) -> i1.compareTo(i2));
        
        //Create result array with number of
        //occurences for each label, plus distances
        Vector res = Vector.zeros(noCategories);
        Vector dist = Vector.zeros(noCategories);
        for (int j = 0; j < settings.K; j++)
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
