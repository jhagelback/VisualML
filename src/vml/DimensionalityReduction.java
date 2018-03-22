
package vml;

import java.text.DecimalFormat;
import java.io.*;

/**
 * Reduces the size (number of attributes) for a dataset using Dimensionality
 * Reduction.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class DimensionalityReduction 
{
    /** Path to dataset file */
    private String filename;
    /** The dataset */
    private Dataset data;
    /** Number of columns to keep in the dataset */
    private int columns;
    /** Reduced dataset */
    private Matrix red;
    /** Type of dr: PCA or SVD */
    private String type;
    
    //For output
    private DecimalFormat df = new DecimalFormat("0.0000");
    
    public static DimensionalityReduction getPCA(String filename, int columns)
    {
        return new DimensionalityReduction(filename, columns, "PCA");
    }
    
    public static DimensionalityReduction getSVD(String filename, int columns)
    {
        return new DimensionalityReduction(filename, columns, "SVD");
    }
    
    /**
     * Initialises a new dimensionality reduction. 
     * 
     * @param filename Path to dataset file
     * @param columns Number of columns to keep in the reduced dataset
     * @param type Type of dimensionality reduction: PCA or SVD
     */
    private DimensionalityReduction(String filename, int columns, String type)
    {
        this.filename = filename;
        this.columns = columns;
        this.type = type;
        
        DataSource reader = new DataSource();
        data = reader.read(filename);
        
        if (data == null)
        {
            System.err.println("Dataset '" + filename + "' was not found!");
            System.exit(1);
        }
    }
    
    /**
     * Reduces the dataset with dimensionality reduction and saves the result.
     */
    public void reduceAndSave()
    {
        if (type.equalsIgnoreCase("PCA")) reducePCA();
        if (type.equalsIgnoreCase("SVD")) reduceSVD();
        saveReducedData();
    }
    
    /**
     * Reduces the dataset with Principal-Component Analysis (PCA).
     */
    public void reducePCA()
    {
        System.out.print("Reducing data with PCA ... ");
        PCA pca = new PCA(data.input_matrix());
        red = pca.analyze(columns);
        System.out.println("done");
    }
    
    /**
     * Reduces the dataset with Singular-Value Decomposition (SVD).
     */
    public void reduceSVD()
    {
        System.out.print("Reducing data with SVD ... ");
        SVD svd = new SVD(data.input_matrix());
        red = svd.analyze();
        System.out.println("done");
    }
    
    /**
     * Saves the reduced dataset to a new data file.
     */
    public void saveReducedData()
    {
        //Output filename
        String out_filename = filename.replaceAll(".csv", "_" + type.toLowerCase() + ".csv");
        System.out.print("Saving reduced dataset to '" + out_filename + "' ... ");
        //Labels vector
        Vector y = data.label_vector();
        
        //Create reduced data file
        try (BufferedWriter out = new BufferedWriter(new FileWriter(out_filename))) 
        {
            //Header line
            String header = "";
            for (int c = 0; c < red.columns(); c++)
            {
                header += "dr" + c + ",";
            }
            header += "category";
            //Write to file
            out.write(header + "\n");

            //Iterate over all data instances
            for (int r = 0; r < red.rows(); r++)
            {
                String str = "";
                for (int c = 0; c < red.columns(); c++)
                {
                    str += df.format(red.get(r, c)) + ",";
                }
                str += data.getCategoryLabel((int)y.get(r));

                //Write to file
                out.write(str + "\n");
            }
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
            System.exit(1);
        }
        
        System.out.println("done");
    }
}
