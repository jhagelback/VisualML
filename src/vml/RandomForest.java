
package vml;

import java.util.ArrayList;
import java.util.Random;

/**
 * Random Forest classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class RandomForest extends Classifier
{
    //Configuration settings
    private RFSettings settings;
    //Forest
    private ArrayList<CART> forest;
    //Randomizer
    private Random rnd;
    
    /**
     * Creates a classifier.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param settings Configuration settings for this classifier
     */
    public RandomForest(Dataset data, Dataset test, RFSettings settings)
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
        o.appendText("Random Forest Classifier");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        
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
        initForest();
        
        return 0;
    }
    
    /**
     * Creates the CART tree forest.
     */
    private void initForest()
    {
        //Randomizer
        rnd = new Random(seed);
        forest = new ArrayList<>();
        
        //Disable output for training the trees
        Logger l = Logger.getConsoleLogger();
        l.disable();
        
        //Init trees
        for (int i = 0; i < settings.trees; i++)
        {
            Dataset d = getRandomSubset(settings.sample_size);
            CART c = new CART(d, test, settings.getTreeSettings());
            c.enableForestRandomizer(seed + i);
            forest.add(c);
        }
        
        //Parallell training
        forest.stream().parallel().forEach((c) -> 
        {
            c.train(l);
        });
    }
    
    /**
     * Returns a random subset of the dataset.
     * 
     * @param sample_size Sample size
     * @return Random subset
     */
    private Dataset getRandomSubset(double sample_size)
    {
        int size = (int)(data.size() * sample_size);
        
        Dataset sub = data.clone_empty();
        for (int i = 0; i < size; i++)
        {
            int rndI = rnd.nextInt(data.size());
            sub.add(data.get(rndI));
        }
        
        return sub;
    }
    
    /**
     * Performs activation for the specified dataset.
     * 
     * @param test Test dataset
     */
    @Override
    public void activation(Dataset test)
    {
        for (int i = 0; i < forest.size(); i++)
        {
            forest.get(i).activation(test);
        }
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
        //Hard voting
        Vector pred = Vector.zeros(data.noCategories());
        for (int c = 0; c < forest.size(); c++)
        {
            int pred_label = forest.get(c).classify(i);
            pred.v[pred_label]++;
        }
        
        return pred.argmax();
    }
}
