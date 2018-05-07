
package vml;

import java.io.File;
import java.text.NumberFormat;
import java.util.*;
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
    //Last modified time for xml file
    private static long last_mod;
    
    /**
     * Builds the classifier with the specified experiment id.
     * 
     * @param id Experiment id
     * @param o Logger for log info
     * @return Classifier, or null if experiments was not found
     */
    public static Classifier build(String id, Logger o)
    {
        try
        {
            return readSettings(id);
        }
        catch (Exception ex)
        {
            o.appendError(ex.getMessage());
        }
        return null;
    }
    
    /**
     * Reads the training dataset from file. The dataset must be of CSV type and be located in
     * the data folder. The application exits if the dataset cannot be found.
     * 
     * @param dataset_name Name of the training dataset
     * @param reader Data reader instance
     * @return The dataset
     * @throws java.lang.Exception If unable to read dataset file or file is not supported
     */
    public static Dataset readDataset(String dataset_name, DataSource reader) throws Exception
    {
        try
        {
            //Check if dataset is found
            if (dataset_name == null) return null;

            //Read data
            Dataset data = reader.read(dataset_name);
 
            return data;
        }
        catch (Exception ex)
        {
            throw ex;
        }
    }
    
    /**
     * Reads the experiments.xml file and searches for the experiment with
     * the specified id.
     * 
     * @param id Experiment id
     * @return Classifier for this experiment, or null if experiment was not found
     * @throws java.lang.Exception If unable to read settings or dataset
     */
    private static Classifier readSettings(String id) throws Exception
    {
        Classifier c = null;
        
        try
        {
            openXML();
            
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
                        if (ctype.equalsIgnoreCase("RBF"))
                        {
                            c = readRBF(e);
                        }
                        if (ctype.equalsIgnoreCase("CART"))
                        {
                            c = readCART(e);
                        }
                        if (ctype.equalsIgnoreCase("RF"))
                        {
                            c = readRF(e);
                        }
                    }      
                }
            }
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a Linear classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     * @throws java.lang.Exception If unable to read settings
     */
    private static Classifier readLinear(Element e) throws Exception
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            LSettings settings = new LSettings();
            if (exists(e, "Epochs")) settings.epochs = getInt(e, "Epochs");
            if (exists(e, "LearningRate")) settings.learningrate = getDouble(e, "LearningRate");
            if (exists(e, "StopThreshold")) settings.stop_threshold = getDouble(e, "StopThreshold");
            if (exists(e, "RegularizationStrength")) settings.lambda = getDouble(e, "RegularizationStrength");
            if (exists(e, "UseRegularization")) settings.use_regularization = getBoolean(e, "UseRegularization");
            if (exists(e, "Normalization")) 
            {
                settings.use_normalization = true;
                settings.normalization_bounds = getNormalization(e, "Normalization");
            }
            if (exists(e, "BatchSize")) settings.batch_size = getInt(e, "BatchSize");
            if (exists(e, "ShuffleData")) settings.shuffle = getBoolean(e, "ShuffleData");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (data == null)
            {
                System.out.println("Unable to find training dataset '" + dataset_name + "'");
                System.exit(1);
            }
            if (settings.shuffle)
            {
                data.shuffle();
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);
            
            //Normalize attributes
            if (settings.use_normalization)
            {
                data.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                if (test != null)
                {
                    test.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                }
            }

            //Init classifier
            c = new Linear(data, test, settings);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a CART Tree classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     * @throws java.lang.Exception If unable to read settings
     */
    private static Classifier readCART(Element e) throws Exception
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            CARTSettings settings = new CARTSettings();
            if (exists(e, "MaxDepth")) settings.max_depth = getInt(e, "MaxDepth");
            if (exists(e, "MinSize")) settings.min_size = getInt(e, "MinSize");
            if (exists(e, "ShuffleData")) settings.shuffle = getBoolean(e, "ShuffleData");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (settings.shuffle)
            {
                data.shuffle();
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);
            
            //Init classifier
            c = new CART(data, test, settings);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a RandomForest Tree classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     * @throws java.lang.Exception If unable to read settings
     */
    private static Classifier readRF(Element e) throws Exception
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            RFSettings settings = new RFSettings();
            if (exists(e, "MaxDepth")) settings.max_depth = getInt(e, "MaxDepth");
            if (exists(e, "MinSize")) settings.min_size = getInt(e, "MinSize");
            if (exists(e, "Trees")) settings.trees = getInt(e, "Trees");
            if (exists(e, "SampleSize")) settings.sample_size = getDouble(e, "SampleSize");
            if (exists(e, "ShuffleData")) settings.shuffle = getBoolean(e, "ShuffleData");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (settings.shuffle)
            {
                data.shuffle();
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);
            
            //Init classifier
            c = new RandomForest(data, test, settings);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a Neural Network classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     * @throws java.lang.Exception If unable to read settings
     */
    private static Classifier readNN(Element e) throws Exception
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            NNSettings settings = new NNSettings();
            if (exists(e, "Epochs")) settings.epochs = getInt(e, "Epochs");
            if (exists(e, "LearningRate")) settings.learningrate = getDouble(e, "LearningRate");
            if (exists(e, "StopThreshold")) settings.stop_threshold = getDouble(e, "StopThreshold");
            if (exists(e, "RegularizationStrength")) settings.lambda = getDouble(e, "RegularizationStrength");
            if (exists(e, "UseRegularization")) settings.use_regularization = getBoolean(e, "UseRegularization");
            if (exists(e, "Normalization")) 
            {
                settings.use_normalization = true;
                settings.normalization_bounds = getNormalization(e, "Normalization");
            }
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
            if (exists(e, "ShuffleData")) settings.shuffle = getBoolean(e, "ShuffleData");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (settings.shuffle)
            {
                data.shuffle();
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);

            //Normalize attributes
            if (settings.use_normalization)
            {
                data.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                if (test != null)
                {
                    test.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                }
            }

            //Init classifier
            c = new NN(data, test, settings);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a k-Nearest Neighbor classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     * @throws java.lang.Exception If unable to read settings
     */
    private static Classifier readKNN(Element e) throws Exception
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            KNNSettings settings = new KNNSettings();
            if (exists(e, "K")) settings.K = getInt(e, "K");
            if (exists(e, "Normalization")) 
            {
                settings.use_normalization = true;
                settings.normalization_bounds = getNormalization(e, "Normalization");
            }
            if (exists(e, "DistanceMeasure"))
            {
                String t = get(e, "DistanceMeasure");
                if (t.equalsIgnoreCase("L1")) settings.distance_measure = KNNSettings.L1;
                if (t.equalsIgnoreCase("L2")) settings.distance_measure = KNNSettings.L2;
            }
            if (exists(e, "ShuffleData")) settings.shuffle = getBoolean(e, "ShuffleData");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (settings.shuffle)
            {
                data.shuffle();
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);

            //Normalize attributes
            if (settings.use_normalization)
            {
                data.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                if (test != null)
                {
                    test.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                }
            }

            //Init classifier
            c = new KNN(data, test, settings);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Read settings and creates a RBF Kernel classifier.
     * 
     * @param e Experiment xml node
     * @return The classifier
     * @throws java.lang.Exception If unable to read settings
     */
    private static Classifier readRBF(Element e) throws Exception
    {
        Classifier c = null;
        
        try
        {
            String dataset_name = get(e, "TrainingData");
            String testset_name = get(e, "TestData");
            
            //Read settings
            KernelSettings settings = new KernelSettings();
            if (exists(e, "Gamma")) settings.gamma = getDouble(e, "Gamma");
            if (exists(e, "Normalization")) 
            {
                settings.use_normalization = true;
                settings.normalization_bounds = getNormalization(e, "Normalization");
            }
            if (exists(e, "ShuffleData")) settings.shuffle = getBoolean(e, "ShuffleData");
            
            //Read training dataset
            DataSource reader = new DataSource();
            Dataset data = ClassifierFactory.readDataset(dataset_name, reader);
            if (settings.shuffle)
            {
                data.shuffle();
            }
            //Read test dataset
            Dataset test = ClassifierFactory.readDataset(testset_name, reader);

            //Normalize attributes
            if (settings.use_normalization)
            {
                data.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                if (test != null)
                {
                    test.normalizeAttributes(settings.normalization_bounds[0], settings.normalization_bounds[1]);
                }
            }

            //Init classifier
            c = new Kernel(data, test, settings);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        
        return c;
    }
    
    /**
     * Returns a list of all experiments in the experiments.xml file.
     * 
     * @return List of experiments
     */
    static ArrayList<ChoiceItem> findAvailableExperiments()
    {
        ArrayList<ChoiceItem> exp = new ArrayList<>();
        
        try
        {
            openXML();
            
            //Search for experiment nodes
            NodeList nodes = root.getElementsByTagName("Experiment");
            for (int n = 0; n < nodes.getLength(); n++)
            {
                Node node = nodes.item(n);
                //Convert from Node to Element
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element e = (Element) node;
                    //Find experiment id
                    String cid = e.getAttribute("id");
                    //Find type and filename
                    String ctype = e.getElementsByTagName("Classifier").item(0).getTextContent();
                    String cdata = e.getElementsByTagName("TrainingData").item(0).getTextContent();
                    cdata = cdata.substring(cdata.lastIndexOf("/")+1, cdata.length());
                    //cdata = cdata.substring(cdata.lastIndexOf("/")+1, cdata.lastIndexOf("."));
                    cdata = cdata.substring(cdata.lastIndexOf("/")+1, cdata.length());
                    exp.add(new ChoiceItem(cid, ctype, cdata));
                }
            }
        }
        catch (Exception ex)
        {
            //Return what has been found so far
        }
        
        return exp;
    }
    
    /**
     * Reads the settings.xml file into memory.
     * 
     * @throws java.lang.Exception If unable to open settings xml file
     */
    private static void openXML() throws Exception
    {
        //Read xml file, if not already read into memory
        try
        {
            //Check if file exists and if it has been modified
            File xml = new File("experiments.xml");
            if (!xml.exists())
            {
                System.err.println("Experiments XML file does not exist");
                System.exit(1);
            }
            //Last modified
            long lm = xml.lastModified();
            if (lm > last_mod)
            {
                //File has been modified. Reload xml
                last_mod = lm;
                root = null;
            }
            
            if (root == null)
            {
                DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
                DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
                Document doc = dBuilder.parse(xml);
                root = doc.getDocumentElement();
            }
        }
        catch (Exception ex)
        {
            throw ex;
        }
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
     * Returns the normalization bounds for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for (NormalizationType)
     * @return Normalization bounds (min and max value)
     * @throws java.lang.Exception If unable to read settings
     */
    private static int[] getNormalization(Element e, String node) throws Exception
    {
        int[] bounds = new int[2];
        
        try
        {
            if (e.getElementsByTagName(node).getLength() == 1)
            {
                String str = e.getElementsByTagName(node).item(0).getTextContent();
                String[] t = str.split(":");
                bounds[0] = Integer.parseInt(t[0]);
                bounds[1] = Integer.parseInt(t[1]);
            }
        }
        catch (Exception ex)
        {
            throw new Exception("Invalid normalization parameter");
        }
        
        return bounds;
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
     * @throws java.lang.Exception If unable to parse integer parameter
     */
    private static int getInt(Element e, String node) throws Exception
    {
        try
        {
            if (e.getElementsByTagName(node).getLength() == 1)
            {
                return Integer.parseInt(e.getElementsByTagName(node).item(0).getTextContent());
            }
        }
        catch (Exception ex)
        {
            throw new Exception("Invalid integer parameter");
        }
        return 0;
    }
    
    /**
     * Returns the decimal value contents for a child node in this element.
     * 
     * @param e Current element
     * @param node Child node to search for
     * @return Double contents, or 0 if not found
     * @throws java.lang.Exception If unable to parse double parameter
     */
    private static double getDouble(Element e, String node) throws Exception
    {
        try
        {
            if (e.getElementsByTagName(node).getLength() == 1)
            {
                String t = e.getElementsByTagName(node).item(0).getTextContent();
                NumberFormat nf = NumberFormat.getInstance(Locale.ENGLISH);
                return nf.parse(t).doubleValue();
            }
        }
        catch (Exception ex)
        {
            throw new Exception("Invalid double parameter");
        }
        return 0;
    }
}
