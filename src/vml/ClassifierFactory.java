
package vml;

import java.io.File;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Factory class for creating classifiers.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class ClassifierFactory 
{
    //experiments.xml root node
    private static Element root;
    
    /**
     * Builds the classifier with the specified experiment id.
     * 
     * @param id Experiment id
     * @return Classifier, or null if experiments was not found
     */
    public static Classifier build(String id)
    {
        return readSettings(id);
    }
    
    /**
     * Reads the training dataset from file. The dataset must be of CSV type and be located in
     * the data folder. The application exits if the dataset cannot be found.
     * 
     * @param dataset_name Name of the training dataset
     * @param reader Data reader instance
     * @return The dataset
     */
    public static Dataset readDataset(String dataset_name, DataSource reader)
    {
        //Check if dataset is found
        if (dataset_name == null) return null;
        String fname = dataset_name;
        if (!fname.endsWith(".csv")) fname += ".csv";
        File f = new File(fname);
        if (!f.exists()) return null;
        
        //Read data
        Dataset data = reader.read(fname);
        
        return data;
    }
    
    /**
     * Reads the experiments.xml file and searches for the experiment with
     * the specified id.
     * 
     * @param id Experiment id
     * @return Classifier for this experiment, or null if experiment was not found
     */
    public static Classifier readSettings(String id)
    {
        Classifier c = null;
        
        try
        {
            //Read xml file, if not already read into memory
            if (root == null)
            {
                File xml = new File("experiments.xml");
                if (!xml.exists())
                {
                    System.err.println("Experiments XML file does not exist");
                    System.exit(1);
                }
                DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
                DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
                Document doc = dBuilder.parse(xml);
                root = doc.getDocumentElement();
            }
            
            //Search for experiment nodes
            NodeList nodes = root.getElementsByTagName("Experiment");
            for (int n = 0; n < nodes.getLength(); n++)
            {
                Node node = nodes.item(n);
                //Convert from Node to Element
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element e = (Element) node;
                    
                    //Check if current node matches the experiment we are looking for
                    String cid = e.getAttribute("id");
                    if (cid.equalsIgnoreCase(id))
                    {
                        //Create classifier of correct type
                        String ctype = e.getElementsByTagName("Classifier").item(0).getTextContent();
                        if (ctype.equalsIgnoreCase("Linear"))
                        {
                            c = readLinear(e);
                        }
                        if (ctype.equalsIgnoreCase("NN"))
                        {
                            c = readNN(e);
                        }
                        if (ctype.equalsIgnoreCase("KNN"))
                        {
                            c = readKNN(e);
                        }
                    }      
                }
            }
             
        }
        catch (Exception ex)
        {
            System.err.println("Experiments XML file is invalid");
            ex.printStackTrace();
            System.exit(1);
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a Linear classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     */
    public static Classifier readLinear(Element e)
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            LSettings settings = new LSettings();
            if (exists(e, "Iterations")) settings.iterations = getInt(e, "Iterations");
            if (exists(e, "LearningRate")) settings.learningrate = getDouble(e, "LearningRate");
            if (exists(e, "RegularizationStrength")) settings.lambda = getDouble(e, "RegularizationStrength");
            if (exists(e, "UseRegularization")) settings.use_regularization = getBoolean(e, "UseRegularization");
            if (exists(e, "NormalizationType")) settings.normalization_type = getNormType(e, "NormalizationType");
            if (exists(e, "BatchSize")) settings.batch_size = getInt(e, "BatchSize");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (data == null)
            {
                System.out.println("Unable to find training dataset '" + dataset_name + "'");
                System.exit(1);
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);

            //Normalize attributes
            data.normalizeAttributes(settings.normalization_type);
            if (test != null)
            {
                test.normalizeAttributes(settings.normalization_type);
            }

            //Init classifier
            c = new Linear(data, test, settings);
        }
        catch (Exception ex)
        {
            System.err.println("Experiments XML file is invalid");
            ex.printStackTrace();
            System.exit(1);
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a Neural Network classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     */
    public static Classifier readNN(Element e)
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            NNSettings settings = new NNSettings();
            if (exists(e, "Iterations")) settings.iterations = getInt(e, "Iterations");
            if (exists(e, "LearningRate")) settings.learningrate = getDouble(e, "LearningRate");
            if (exists(e, "RegularizationStrength")) settings.lambda = getDouble(e, "RegularizationStrength");
            if (exists(e, "UseRegularization")) settings.use_regularization = getBoolean(e, "UseRegularization");
            if (exists(e, "NormalizationType")) settings.normalization_type = getNormType(e, "NormalizationType");
            if (exists(e, "UseMomentum")) settings.use_momentum = getBoolean(e, "UseMomentum");
            if (exists(e, "BatchSize")) settings.batch_size = getInt(e, "BatchSize");
            if (exists(e, "HiddenLayers"))
            {
                String[] t = get(e, "HiddenLayers").split(",");
                settings.layers = new int[t.length];
                for (int i = 0; i < t.length; i++)
                {
                    settings.layers[i] = Integer.parseInt(t[i].trim());
                }
            }
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (data == null)
            {
                System.out.println("Unable to find training dataset '" + dataset_name + "'");
                System.exit(1);
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);

            //Normalize attributes
            data.normalizeAttributes(settings.normalization_type);
            if (test != null)
            {
                test.normalizeAttributes(settings.normalization_type);
            }

            //Init classifier
            c = new NN(data, test, settings);
        }
        catch (Exception ex)
        {
            System.err.println("Experiments XML file is invalid");
            System.exit(1);
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a k-Nearest Neighbor classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     */
    public static Classifier readKNN(Element e)
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            KNNSettings settings = new KNNSettings();
            if (exists(e, "K")) settings.K = getInt(e, "K");
            if (exists(e, "NormalizationType")) settings.normalization_type = getNormType(e, "NormalizationType");
            if (exists(e, "DistanceMeasure"))
            {
                String t = get(e, "DistanceMeasure");
                if (t.equalsIgnoreCase("L1")) settings.distance_measure = KNNSettings.L1;
                if (t.equalsIgnoreCase("L2")) settings.distance_measure = KNNSettings.L2;
            }
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (data == null)
            {
                System.out.println("Unable to find training dataset '" + dataset_name + "'");
                System.exit(1);
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);

            //Normalize attributes
            data.normalizeAttributes(settings.normalization_type);
            if (test != null)
            {
                test.normalizeAttributes(settings.normalization_type);
            }

            //Init classifier
            c = new KNN(data, test, settings);
        }
        catch (Exception ex)
        {
            System.err.println("Experiments XML file is invalid");
            System.exit(1);
        }
        
        return c;
    }
    
    /**
     * Checks if a child node exists in this element.
     * 
     * @param e Current element
     * @param node Child node to search for
     * @return True if exists, false otherwise
     */
    private static boolean exists(Element e, String node)
    {
        return e.getElementsByTagName(node).getLength() != 0;
    }
    
    /**
     * Returns the text contents for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for
     * @return Text contents, or null if not found
     */
    private static String get(Element e, String node)
    {
        if (e.getElementsByTagName(node).getLength() == 1)
        {
            return e.getElementsByTagName(node).item(0).getTextContent();
        }
        return null;
    }
    
    /**
     * Returns the normalization type for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for (NormalizationType)
     * @return Normalization type (None, NegPos or Pos)
     */
    private static int getNormType(Element e, String node)
    {
        if (e.getElementsByTagName(node).getLength() == 1)
        {
            String t = e.getElementsByTagName(node).item(0).getTextContent();
            if (t.equalsIgnoreCase("none")) return Dataset.Norm_NONE;
            if (t.equalsIgnoreCase("NegPos")) return Dataset.Norm_NEGPOS;
            if (t.equalsIgnoreCase("Pos")) return Dataset.Norm_POS;
        }
        return Dataset.Norm_NONE;
    }
    
    /**
     * Returns the boolean contents for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for
     * @return Boolean contents (true or false)
     */
    private static boolean getBoolean(Element e, String node)
    {
        if (e.getElementsByTagName(node).getLength() == 1)
        {
            String t = e.getElementsByTagName(node).item(0).getTextContent();
            return t.equalsIgnoreCase("true");
        }
        return false;
    }
    
    /**
     * Returns the integer contents for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for
     * @return Integer contents, or 0 if not found
     */
    private static int getInt(Element e, String node) throws NumberFormatException
    {
        if (e.getElementsByTagName(node).getLength() == 1)
        {
            return Integer.parseInt(e.getElementsByTagName(node).item(0).getTextContent());
        }
        return 0;
    }
    
    /**
     * Returns the decimal value contents for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for
     * @return Double contents, or 0 if not found
     */
    private static double getDouble(Element e, String node) throws ParseException
    {
        if (e.getElementsByTagName(node).getLength() == 1)
        {
            String t = e.getElementsByTagName(node).item(0).getTextContent();
            NumberFormat nf = NumberFormat.getInstance(Locale.ENGLISH);
            return nf.parse(t).doubleValue();
        }
        return 0;
    }
}
