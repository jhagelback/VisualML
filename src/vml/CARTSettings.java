
package vml;

/**
 * Contains settings for the CART classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class CARTSettings 
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
     * Creates default settings.
     */
    public CARTSettings()
    {
        max_depth = 5;
        min_size = 10;
        shuffle = true;
    }
}
