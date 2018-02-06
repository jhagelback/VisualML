
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
    int K = 3;
    
    /**
     * Normalization type to use for the data.
     */
    int normalization_type = Dataset.Norm_NONE;
    
    /**
     * Distance measure L1 (Manhattan distance)
     */
    static final int L1 = 1;
    
    /**
     * Distance measure L2 (Euclidean distance)
     */
    static final int L2 = 2;
    
    /**
     * Distance measure (L1 or L2) to use.
     */
    int distance_measure = L2;
    
    /**
     * Creates default settings.
     */
    public KNNSettings()
    {
        K = 3;
        distance_measure = L2;
        normalization_type = Dataset.Norm_NONE;
    }
}
