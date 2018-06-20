
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
    private Tensor2D data;
    
    /**
     * Initialises a new SVD for a dataset.
     * 
     * @param data The dataset
     */
    public SVD(Tensor2D data)
    {
        //Transpose the data since we have instances as columns instead of rows
        this.data = Tensor2D.transpose(data);
    }
    
    /**
     * Executes SVD and returns a reduced dataset.
     * 
     * @return Reduced dataset
     */
    public Tensor2D analyze()
    {
        //Calculate M^TM
        Tensor2D mTm = Tensor2D.transpose_mul(data, data);
        //Calculate MM^M
        Tensor2D mmT = Tensor2D.mul_transpose(data, data);
        //Calculate Eigenpairs
        EigenDecomp edV = new EigenDecomp(mTm);
        edV.decomp();
        EigenDecomp edU = new EigenDecomp(mmT);
        edU.decomp();
        
        //Create U and V
        Tensor2D U = edU.E;
        Tensor2D V = Tensor2D.transpose(edV.E);
        //Create Sigma matrix (diagonal is square root of eigenvalues)
        Tensor1D EV = edV.EV;
        Tensor2D S = Tensor2D.zeros(EV.size(), EV.size());
        for (int i = 0; i < EV.size(); i++)
        {
            S.v[i][i] = Math.sqrt(Math.abs(EV.v[i]));
        }
        
        //Find number of concepts to remove
        int c = reduce_concepts(S);
        //Remove concepts
        U = Tensor2D.sub(U, U.rows(), c);
        V = Tensor2D.sub(V, c, V.columns());
        S = Tensor2D.sub(S, c, c);
        
        //Orignal data matrix can be reconstructed
        //(not needed for the transform)
        //Tensor2D B = Tensor2D.mul(U, S);
        //B = Tensor2D.mul(B, V);
        
        //Reduce dimensionality
        Tensor2D T = Tensor2D.mul(U, S);
        
        return T;
    }
    
    /**
     * Checks how many concepts we can remove, returning the
     * number of columns to keep in the matrix.
     * 
     * @param S Sigma matrix
     * @return Number of columns to keep
     */
    private int reduce_concepts(Tensor2D S)
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
