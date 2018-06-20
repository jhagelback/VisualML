
package vml;

/**
 * Principal-Component Analysis (PCA) dimensionality reduction used to reduce the 
 * number of attributes in a dataset.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class PCA 
{
    /** Input data */
    private Tensor2D data;
    
    /**
     * Initialises a new PCA for a dataset.
     * 
     * @param data The dataset
     */
    public PCA(Tensor2D data)
    {
        //Transpose the data since we have instances as columns instead of rows
        this.data = Tensor2D.transpose(data);
    }
    
    /**
     * Executes PCA and returns a reduced dataset.
     * 
     * @param columns Number of columns to keep
     * @return Reduced dataset
     */
    public Tensor2D analyze(int columns)
    {
        //Calculate M^TM
        Tensor2D mTm = Tensor2D.transpose_mul(data, data);
        //Calculate Eigenpairs
        EigenDecomp ed = new EigenDecomp(mTm);
        ed.decomp();
        //Transform the dataset
        Tensor2D ME = Tensor2D.mul(data, ed.E);
        //Reduce the size of dataset by removing columns
        Tensor2D red = Tensor2D.sub(ME, data.rows(), columns);
        return red;
    }
}
