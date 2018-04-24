
package vml;

import java.io.*;
import java.util.HashMap;

import java.util.zip.*;

/**
 * Reads comma-separated values (csv) data files into a dataset container. Supports compressed
 * (zip) files containing csv dataset files.
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
     * @throws java.lang.Exception If unable to read dataset file
     */
    public Dataset read(String filename) throws Exception
    {
        try
        {
            //Dataset file
            File f = new File(filename);
            Dataset dset = new Dataset(f.getName());
        
            //Open reader
            BufferedReader in = open(f);
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
            
            //Set int label to category label
            dset.setLabelMapping(intToCat);

            return dset;
        }
        catch (Exception ex)
        {
            throw ex;
        }
    }
    
    /**
     * Opens a reader to a dataset file. Supports csv and compressed (zip) files.
     * 
     * @param f Dataset file
     * @return File reader
     * @throws Exception If dataset file is not supported
     */
    private BufferedReader open(File f) throws Exception
    {
        try
        {
            //Read CSV file
            if (f.getName().endsWith(".csv"))
            {
                return new BufferedReader(new FileReader(f));
            }
            //Read ZIP file
            else if(f.getName().endsWith(".zip"))
            {
                ZipFile zf = new ZipFile(f);
                while (zf.entries().hasMoreElements())
                {
                    ZipEntry e = zf.entries().nextElement();
                    if (e.getName().endsWith(".csv"))
                    {
                        return new BufferedReader(new InputStreamReader(zf.getInputStream(e)));
                    }
                }
                throw new Exception("Zip file does not contain a csv file");
            }
            else
            {
                throw new Exception("Not a valid dataset file: " + f.getName());
            }
        }
        catch (Exception ex)
        {
            throw new Exception("Unable to find dataset file");
        }
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
