
package vml;

/**
 * Eigendecomposition into Eigenpairs (Eigenvector and Eigenvalue) of a matrix.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class EigenDecomp 
{
    /** The dataset matrix */
    private Matrix data;
    /** Eigenvectors */
    protected Matrix E;
    /** Eigenvalues */
    protected Vector EV;
    /** Used in Power Iteration */
    private Matrix cM;
    
    /**
     * Intialises a new Eigenpairs decomposition.
     * 
     * @param data The input data
     */
    public EigenDecomp(Matrix data)
    {
        //Temp
        /*double[][] v = {
            {3, 2},
            {2, 6}
        };
        m = new Matrix(v);//*/
        
        /*double[][] v = {
            {5, 4, 11, 10},
            {4, 5, 10, 11},
            {11, 10, 25, 24},
            {10, 11, 24, 25}
        };
        m = new Matrix(v);//*/
        
        this.data = data;
        
        //Create a copy of matrix M
        cM = data.copy();
        
        //Placeholders for Eigenvectors and Eigenvalues
        E = Matrix.zeros(data.rows(), data.columns());
        EV = Vector.zeros(data.columns());     
    }
    
    /**
     * Decomposed the data into Eigenpairs.
     */
    public void decomp()
    {
        //Calculate eigenpairs for all columns in the input data
        for (int c = 0; c < data.columns(); c++)
        {
            next(c);
        }
        
        System.out.println("Eigenvalues are:");
        System.out.println(EV);
        
        //Verify that the result is correct
        if (!checkResult())
        {
            System.err.println("Eigenvectors might be invalid. Double-check result.");
        }
    }
    
    /**
     * Checks the result of the Eigenvectors decomposition. If everything is correct, the
     * multiplication between the Eigenvectors matrix and its transpose shall be the
     * identity matrix (1's in the main diagonal, 0's otherwise).
     * 
     * @return True if the decomposition is correct, false otherwise
     */
    private boolean checkResult()
    {
        //Calculate the product between the Eigenvectors matrix
        //and its transpose
        Matrix I = Matrix.transpose_mul(E, E);
        
        //Check values
        for (int r = 0; r < I.rows(); r++)
        {
            for (int c = 0; c < I.columns(); c++)
            {
                //Current value
                double val = I.v[r][c];
                //Should be 0, unless...
                double targ = 0;
                //... it is the main diagonal, then it should be 1
                if (r == c) targ = 1;
                //If below threshold, there might be something wrong with
                //the decomposition.
                if (Math.abs(val - targ) > 0.01) return false;
            }
        }
        
        //Everything is Ok
        return true;
    }
    
    /**
     * Calculates the next Eigenpair for the input data matrix.
     * 
     * @param c Column number
     */
    private void next(int c)
    {
        //Create a random x vector to decrease the chance that the vector
        //is orthogonal to the eigenvector (the result will then be incorrect.
        Vector x = Vector.random_norm(cM.rows());
        
        //When to stop iteration
        double diff = 1.0;
        int it = 0;
        
        //Power Iteration to find Eigenvector
        while (diff > 0.00001 && it <= 50)
        {
            //Multiply M with x0
            Vector x1 = Matrix.mul(cM,x);
            //Calculate Frobenius norm
            double n = x1.frobenius_norm();
            //Divide x with the norm
            x1.divide(n);
            //Calculate the difference between current and previous x
            //(square root of the squared sum of the difference between the components)
            diff = Vector.diff(x, x1);
            //Set new x
            x = x1;
            //Next iteration
            it++;
        }
        
        //Calculate Eigenvalue
        Vector v = Matrix.transpose_mul(x, cM);
        double ev = v.dot(x);
        
        //Copy to placeholders
        E.insert(x, c);
        EV.v[c] = ev;
        
        //Create new M for next eigenpair
        Matrix xxT = Vector.mul(x, x);
        xxT.multiply(ev);
        cM.subtract(xxT);
    }  
}
