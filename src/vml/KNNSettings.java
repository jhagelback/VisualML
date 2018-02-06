
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
     * Creates default settings.
     */
    public KNNSettings()
    {
        K = 3;
        normalization_type = Dataset.Norm_NONE;
    }
}
