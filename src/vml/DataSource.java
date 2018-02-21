
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
    /**
     * Seed for dataset shuffle randomiser.
     */
    public static int seed = 7;
    //Conversion from string label to int label
    private HashMap<String,Integer> catToInt;
    //Mapping from int label to string label
    private HashMap<Integer,String> intToCat;
    
    /**
     * Creates a new data reader.
     */
    public DataSource()
    {
        catToInt = new HashMap<>();
        intToCat = new HashMap<>();
    }
    
    /**
     * Reads the dataset and returns a dataset container.
     * 
     * @param filename Filename for the data file
     * @return Dataset container
     */
    public Dataset read(String filename)
    {
        File f = new File(filename);
        Dataset dset = new Dataset(f.getName());
        
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
        
        //Set int label to category label
        dset.setLabelMapping(intToCat);
        
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
            String slab = t[t.length - 1];
            //Convert category string to integer
            if (!catToInt.containsKey(slab))
            {
                //Add new conversion
                catToInt.put(slab, catToInt.size());
            }
            //Get category as integer
            int label = catToInt.get(slab);

            //Mapping from category int to string
            if (!intToCat.containsKey(label))
            {
                //Add new mapping
                intToCat.put(label, slab);
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
