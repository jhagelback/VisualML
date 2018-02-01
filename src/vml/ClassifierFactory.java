
package vml;

import java.io.File;

/**
 * Factory class for creating classifiers.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class ClassifierFactory 
{
    /**
     * Creates a Linear Softmax classifier.
     * 
     * @param dataset_name Training dataset
     * @param testset_name Test dataset
     * @param iterations Number of iterations for training
     * @param learningrate Learning rate
     * @param norm_type Type of data normalization
     * @return Linear classifier
     */
    public static Classifier createLinear(String dataset_name, String testset_name, int iterations, double learningrate, int norm_type)
    {
        //Read training dataset
        Dataset data = readDataset(dataset_name);
        if (data == null)
        {
            System.out.println("Unable to find training dataset '" + dataset_name + "'");
            System.exit(0);
        }
        //Read test dataset
        Dataset test = readDataset(testset_name);
        
        //Normalize attributes
        data.normalizeAttributes(norm_type);
        if (test != null)
        {
            test.normalizeAttributes(norm_type);
        }
        
        //Init classifier
        Classifier c = new Linear(data, test, iterations, learningrate);
        
        return c;
    }
    
    /**
     * Creates the demonstration Linear classifier as shown here:
     * http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/
     * 
     * @return Linear classifier
     */
    public static Classifier createLinearDemo()
    {
        //Init classifier
        Classifier c = new Linear();
        
        return c;
    }
    
    /**
     * Creates a Neural Network Softmax classifier.
     * 
     * @param dataset_name Training dataset
     * @param testset_name Test dataset
     * @param size_h Number of hidden units
     * @param iterations Number of iterations for training
     * @param learningrate Learning rate
     * @param norm_type Type of data normalization
     * @return Linear classifier
     */
    public static Classifier createNN(String dataset_name, String testset_name, int size_h, int iterations, double learningrate, int norm_type)
    {
        //Read training dataset
        Dataset data = readDataset(dataset_name);
        if (data == null)
        {
            System.out.println("Unable to find training dataset '" + dataset_name + "'");
            System.exit(0);
        }
        //Read test dataset
        Dataset test = readDataset(testset_name);
        
        //Normalize attributes
        data.normalizeAttributes(norm_type);
        if (test != null)
        {
            test.normalizeAttributes(norm_type);
        }
        
        //Init classifier
        Classifier c = new NN(data, test, size_h, iterations, learningrate);
        
        return c;
    }
    
    /**
     * Creates a Deep Neural Network Softmax classifier.
     * 
     * @param dataset_name Training dataset
     * @param testset_name Test dataset
     * @param size_h1 Number of units in hidden layer 1
     * @param size_h2 Number of units in hidden layer 2
     * @param iterations Number of iterations for training
     * @param learningrate Learning rate
     * @param norm_type Type of data normalization
     * @return Linear classifier
     */
    public static Classifier createDNN(String dataset_name, String testset_name, int size_h1, int size_h2, int iterations, double learningrate, int norm_type)
    {
        //Read training dataset
        Dataset data = readDataset(dataset_name);
        if (data == null)
        {
            System.out.println("Unable to find training dataset '" + dataset_name + "'");
            System.exit(0);
        }
        //Read test dataset
        Dataset test = readDataset(testset_name);
        
        //Normalize attributes
        data.normalizeAttributes(norm_type);
        if (test != null)
        {
            test.normalizeAttributes(norm_type);
        }
        
        //Init classifier
        Classifier c = new DeepNN(data, test, size_h1, size_h2, iterations, learningrate);
        
        return c;
    }
    
    /**
     * Creates a k-Nearest Neighbor classifier.
     * 
     * @param dataset_name Training dataset
     * @param testset_name Test dataset
     * @param K K-value (number of neighbors to evaluate)
     * @param norm_type Type of data normalization
     * @return Linear classifier
     */
    public static Classifier createKNN(String dataset_name, String testset_name, int K, int norm_type)
    {
        //Read training dataset
        Dataset data = readDataset(dataset_name);
        if (data == null)
        {
            System.out.println("Unable to find training dataset '" + dataset_name + "'");
            System.exit(0);
        }
        //Read test dataset
        Dataset test = readDataset(testset_name);
        
        //Normalize attributes
        data.normalizeAttributes(norm_type);
        if (test != null)
        {
            test.normalizeAttributes(norm_type);
        }
        
        //Init classifier
        Classifier c = new KNN(data, test, K);
        
        return c;
    }
    
    /**
     * Reads the training dataset from file. The dataset must be of CSV type and be located in
     * the data folder. The application exits if the dataset cannot be found.
     * 
     * @param dataset_name Name of the training dataset
     * @return The dataset
     */
    public static Dataset readDataset(String dataset_name)
    {
        //Check if dataset is found
        if (dataset_name == null) return null;
        String fname = dataset_name;
        if (!fname.endsWith(".csv")) fname += ".csv";
        File f = new File("data/" + fname);
        if (!f.exists()) return null;
        
        //Read data
        DataSource reader = new DataSource("data/" + fname);
        Dataset data = reader.read();
        
        return data;
    }
}
