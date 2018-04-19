
package vml;

/**
 * Contains settings for the RandomForest classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class RFSettings 
{
    /**
     * Max tree depth.
     */
    public int max_depth = 5;
    
    /**
     * Min size for split.
     */
    public int min_size = 10;
    
    /**
     * Sets if training dataset shall be shuffled or not.
     */
    public boolean shuffle = true;
    
    /**
     * Number of CART trees in the forest.
     */
    public int trees = 7;
    
    /**
     * Sample size of the data subset used for each tree.
     */
    public double sample_size = 0.9;
    
    /**
     * Creates default settings.
     */
    public RFSettings()
    {
        max_depth = 5;
        min_size = 10;
        shuffle = true;
        trees = 7;
        sample_size = 0.9;
    }
    
    /**
     * Returns the settings used for the individual CART trees.
     * 
     * @return CART trees settings
     */
    public CARTSettings getTreeSettings()
    {
        CARTSettings settings = new CARTSettings();
        settings.max_depth = max_depth;
        settings.min_size = min_size;
        return settings;
    }
}
