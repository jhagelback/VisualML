
package vml;

/**
 * Singular-Value Decomposition (SVD) dimensionality reduction used to reduce the 
 * number of attributes in a dataset.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class SVD 
{
    /** Input data */
    private Matrix data;
    
    /**
     * Initialises a new SVD for a dataset.
     * 
     * @param data The dataset
     */
    public SVD(Matrix data)
    {
        //Transpose the data since we have instances as columns instead of rows
        this.data = Matrix.transpose(data);
    }
    
    /**
     * Executes SVD and returns a reduced dataset.
     * 
     * @return Reduced dataset
     */
    public Matrix analyze()
    {
        //Calculate M^TM
        Matrix mTm = Matrix.transpose_mul(data, data);
        //Calculate MM^M
        Matrix mmT = Matrix.mul_transpose(data, data);
        //Calculate Eigenpairs
        EigenDecomp edV = new EigenDecomp(mTm);
        edV.decomp();
        EigenDecomp edU = new EigenDecomp(mmT);
        edU.decomp();
        
        //Create U and V
        Matrix U = edU.E;
        Matrix V = Matrix.transpose(edV.E);
        //Create Sigma matrix (diagonal is square root of eigenvalues)
        Vector EV = edV.EV;
        Matrix S = Matrix.zeros(EV.size(), EV.size());
        for (int i = 0; i < EV.size(); i++)
        {
            S.v[i][i] = Math.sqrt(Math.abs(EV.v[i]));
        }
        
        //Find number of concepts to remove
        int c = reduce_concepts(S);
        //Remove concepts
        U = Matrix.submatrix(U, U.rows(), c);
        V = Matrix.submatrix(V, c, V.columns());
        S = Matrix.submatrix(S, c, c);
        
        //Orignal data matrix can be reconstructed
        //(not needed for the transform)
        //Matrix B = Matrix.mul(U, S);
        //B = Matrix.mul(B, V);
        
        //Reduce dimensionality
        Matrix T = Matrix.mul(U, S);
        
        return T;
    }
    
    /**
     * Checks how many concepts we can remove, returning the
     * number of columns to keep in the matrix.
     * 
     * @param S Sigma matrix
     * @return Number of columns to keep
     */
    private int reduce_concepts(Matrix S)
    {
        //First, calculate absolute sum
        double sum = 0;
        for (int i = 0; i < S.columns(); i++)
        {
            sum += Math.abs(S.v[i][i]);
        }
        
        //Second, find number of concepts to remove
        //Not more than 10% of the energy shall be removed
        double energy = 0;
        int concepts = 1;
        for (int i = 0; i < S.columns(); i++)
        {
            energy += Math.abs(S.v[i][i]);
            if (energy <= sum * 0.9)
            {
                concepts++;
            }
            else
            {
                break;
            }
        }
        
        //Remove at least one concept, otherwise SVC has
        //no effect.
        if (concepts == S.columns())
        {
            concepts--;
        }
        
        return concepts;
    }
}
