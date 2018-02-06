
package vml;

import java.io.*;
import java.util.HashMap;

/**
 * Reads comma-separated values (csv) data files into a dataset container.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class DataSource 
{
    //Filename for the data file
    private String filename;
    //Conversion from string label to int label
    private HashMap<String,Integer> catToInt;
    
    /**
     * Creates a new data reader.
     * 
     * @param filename Filename for the data file
     */
    public DataSource(String filename)
    {
        this.filename = filename;
        catToInt = new HashMap<>();
    }
    
    /**
     * Reads the dataset and returns a dataset container.
     * 
     * @return Dataset container
     */
    public Dataset read()
    {
        Dataset dset = new Dataset();
        
        try
        {
            BufferedReader in = new BufferedReader(new FileReader(filename));
            String str = in.readLine();
            //Skip header line
            str = in.readLine();
            
            while (str != null)
            {
                //Convert row to instance
                Instance inst = toInstance(str);
                //If conversion was ok, add to dataset
                if (inst != null) dset.add(inst);
                //Read next line
                str = in.readLine();
            }
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
        
        return dset;
    }
    
    /**
     * Converts a data file row to an instance.
     * 
     * @param line Data file row
     * @return Instance, or null if failed
     */
    private Instance toInstance(String line)
    {
        //Line is empty
        if (line.trim().equals("")) return null;
        
        try
        {
            //Split by comma
            String[] t = line.split(",");
            
            //Read all values
            double[] v = new double[t.length - 1];
            for (int i = 0; i < t.length - 1; i++)
            {
                v[i] = Double.parseDouble(t[i].trim());
            }
            
            //Label
            int label = 0;
            try
            {
                //Category is already integer
                label = Integer.parseInt(t[t.length - 1]);
            }
            catch (Exception ex)
            {
                String slab = t[t.length - 1];
                //Convert category string to integer
                if (!catToInt.containsKey(slab))
                {
                    //Add new conversion
                    catToInt.put(slab, catToInt.size());
                }
                //Get category as integer
                label = catToInt.get(slab);
            }
            
            //Create instance
            Instance inst = new Instance(v, label);
            return inst;
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
        return null;
    }
}
