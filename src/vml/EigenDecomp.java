
package vml;

import java.util.ArrayList;

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
    private Matrix new_M;
    
    /** Temporary placeholder for eigenvectors */
    private ArrayList<Vector> e_list;
    /** Temporary placeholder for eigenvalues */
    private ArrayList<Double> ev_list;
    
    
    /**
     * Intialises a new Eigenpairs decomposition.
     * 
     * @param data The input data
     */
    public EigenDecomp(Matrix data)
    {
        this.data = data;
        
        //Create a copy of matrix M
        new_M = data.copy();     
    }
    
    /**
     * Decomposed the data into Eigenpairs.
     */
    public void decomp()
    {
        //Temporary placeholders for eigenpairs
        e_list = new ArrayList<>();
        ev_list = new ArrayList<>();
        
        //Calculate all eigenpairs for the input data
        while(next()) {}
        
        //Placeholders for Eigenvectors and Eigenvalues
        E = Matrix.zeros(data.rows(), e_list.size());
        EV = Vector.zeros(e_list.size());
        //Copy to placeholders
        for (int i = 0; i < e_list.size(); i++)
        {
            E.insert(e_list.get(i), i);
            EV.set(i, ev_list.get(i));
        }
        
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
     */
    private boolean next()
    {
        //Create a random x vector to decrease the chance that the vector
        //is orthogonal to the eigenvector (the result will then be incorrect.
        Vector x = Vector.random_norm(new_M.rows());

        //When to stop iteration
        double diff = 1.0;
        int it = 0;
        double n = 1000;
        
        //Power Iteration to find Eigenvector
        while (diff > 0.000005 && it <= 100)
        {
            //Multiply M with x0
            Vector new_x = Matrix.mul(new_M, x);
            //Calculate Frobenius norm
            double new_n = new_x.frobenius_norm();    
            //Divide x with the norm
            new_x.div(new_n);
            //Calculate the absolut difference from previous Frobenius norm
            diff = Math.abs(n - new_n);
            
            //Set new x and n
            n = new_n;
            x = new_x;
            
            //Next iteration
            it++;
        }
        
        //Re-orient the vector so first component is positive
        //This is fine since it creates an orthogonal vector to the eigenvector
        if (x.get(0) < 0)
        {
            x.mul(-1);
        }
        
        //Calculate Eigenvalue
        Vector v = Matrix.transpose_mul(x, new_M);
        double ev = v.dot(x);
        //Stop if eigenvalue is 0
        if (Math.abs(ev) <= 0.0001) return false;
        
        //Copy to temporary placeholders
        e_list.add(x);
        ev_list.add(ev);
        
        //Create new M for next eigenpair
        Matrix xxT = Vector.mul(x, x);
        xxT.multiply(ev);
        new_M.subtract(xxT);
        
        return true;
    }
}
