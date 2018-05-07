
package vml;

/**
 * Scale and shift settings used when visualizing a dataset.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
class VizDatasetSettings 
{
    //Dataset
    String dataset_file = "";
    //Scales
    double scale_x = 1;
    double scale_y = 1;
    //Shifts
    double shift_x = 0;
    double shift_y = 0;
    
    /**
     * Creates new settings for a dataset.
     * 
     * @param dataset_file Dataset filename
     * @param scale_x X-scale
     * @param scale_y Y-scale
     * @param shift_x X-shift
     * @param shift_y Y-shift
     */
    public VizDatasetSettings(String dataset_file, double scale_x, double scale_y, double shift_x, double shift_y)
    {
        this.dataset_file = dataset_file;
        this.scale_x = scale_x;
        this.scale_y = scale_y;
        this.shift_x = shift_x;
        this.shift_y = shift_y;
    }
}
