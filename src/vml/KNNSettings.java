
package vml;

/**
 * Contains settings for the k-Nearest Neighbor classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class KNNSettings 
{
    /**
     * K-value (number of near neighbors to evaluate).
     */
    public int K = 3;
    
    /**
     * Sets if data shall be normalized.
     */
    public boolean use_normalization = false;
    
    /**
     * Sets lower and upper bounds for normalized values.
     */
    public int[] normalization_bounds = new int[2];
    
    /**
     * Distance measure L1 (Manhattan distance)
     */
    public static final int L1 = 1;
    
    /**
     * Distance measure L2 (Euclidean distance)
     */
    public static final int L2 = 2;
    
    /**
     * Distance measure (L1 or L2) to use.
     */
    public int distance_measure = L2;
    
    /**
     * Sets if training dataset shall be shuffled or not.
     */
    public boolean shuffle = true;
    
    /**
     * Creates default settings.
     */
    public KNNSettings()
    {
        K = 3;
        distance_measure = L2;
        use_normalization = false;
        normalization_bounds = new int[2];
        shuffle = true;
    }
}
