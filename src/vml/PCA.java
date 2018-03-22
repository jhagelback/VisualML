
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
    private Matrix data;
    
    /**
     * Initialises a new PCA for a dataset.
     * 
     * @param data The dataset
     */
    public PCA(Matrix data)
    {
        //Transpose the data since we have instances as columns instead of rows
        this.data = Matrix.transpose(data);
    }
    
    /**
     * Executes PCA and returns a reduced dataset.
     * 
     * @param columns Number of columns to keep
     * @return Reduced dataset
     */
    public Matrix analyze(int columns)
    {
        //Calculate M^TM
        Matrix mTm = Matrix.transpose_mul(data, data);
        //Calculate Eigenpairs
        EigenDecomp ed = new EigenDecomp(mTm);
        ed.decomp();
        //Transform the dataset
        Matrix ME = Matrix.mul(data, ed.E);
        //Reduce the size of dataset by removing columns
        Matrix red = Matrix.submatrix(ME, data.rows(), columns);
        return red;
    }
}
