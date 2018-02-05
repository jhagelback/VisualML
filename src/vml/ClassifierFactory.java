
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
     * @param settings Configuration settings
     * @return Linear classifier
     */
    public static Classifier createLinear(String dataset_name, String testset_name, LSettings settings)
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
        data.normalizeAttributes(settings.normalization_type);
        if (test != null)
        {
            test.normalizeAttributes(settings.normalization_type);
        }
        
        //Init classifier
        Classifier c = new Linear(data, test, settings);
        
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
     * Creates a Neural Network Softmax classifier with one or more hidden layers.
     * 
     * @param dataset_name Training dataset
     * @param testset_name Test dataset
     * @param settings Configuration settings
     * @return Neural Network classifier
     */
    public static Classifier createNN(String dataset_name, String testset_name, NNSettings settings)
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
        data.normalizeAttributes(settings.normalization_type);
        if (test != null)
        {
            test.normalizeAttributes(settings.normalization_type);
        }
        
        //Init classifier
        Classifier c = new NN(data, test, settings);
        
        return c;
    }
    
    /**
     * Creates a k-Nearest Neighbor classifier.
     * 
     * @param dataset_name Training dataset
     * @param testset_name Test dataset
     * @param settings Configuration settings
     * @return kNN classifier
     */
    public static Classifier createKNN(String dataset_name, String testset_name, KNNSettings settings)
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
        data.normalizeAttributes(settings.normalization_type);
        if (test != null)
        {
            test.normalizeAttributes(settings.normalization_type);
        }
        
        //Init classifier
        Classifier c = new KNN(data, test, settings);
        
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
    
    /**
     * Returns the settings to use for Linear classifiers on the supplied datasets.
     * 
     * @param file Dataset identifier
     * @return Settings to use
     */
    public static LSettings getLSettings(String file)
    {
        LSettings s = null;
        
        switch(file)
        {
            case "demo":
                s = new LSettings();
                s.iterations = 10;
                s.learningrate = 1.0;
                break;
            case "iris":
                s = new LSettings();
                s.iterations = 300;
                s.learningrate = 0.1;
                break;
            case "iris.2d":
                s = new LSettings();
                s.iterations = 50;
                s.learningrate = 1.0;
                s.normalization_type = Dataset.Norm_NEGPOS;
                break;
            case "iris_test":
                s = new LSettings();
                s.iterations = 300;
                s.learningrate = 0.1;
                break;
            case "spiral":
                s = new LSettings();
                s.iterations = 200;
                s.learningrate = 0.1;
                break;
            case "diabetes":
                s = new LSettings();
                s.iterations = 40;
                s.learningrate = 1.0;
                s.normalization_type = Dataset.Norm_NEGPOS;
                break;
            case "circle":
                s = new LSettings();
                s.iterations = 20;
                s.learningrate = 1.0;
                break;
            case "glass":
                s = new LSettings();
                s.iterations = 50;
                s.learningrate = 1.0;
                s.normalization_type = Dataset.Norm_NEGPOS;
                break;
            case "mnist":
                s = new LSettings();
                s.iterations = 200;
                s.learningrate = 1.0;
                s.normalization_type = Dataset.Norm_POS;
                break;
        }
        
        return s;
    }
    
    /**
     * Returns the settings to use for Neural Network classifiers on the supplied datasets.
     * 
     * @param type Classifier type (nn or dnn)
     * @param file Dataset identifier
     * @return Settings to use
     */
    public static NNSettings getNNSettings(String type, String file)
    {
        NNSettings s = null;
        
        if (type.equalsIgnoreCase("nn"))
        {
            /**
             * Neural Network classifiers 
             */
            switch (file) 
            {
                case "demo":
                    s = new NNSettings();
                    s.layers = new int[]{8};
                    s.iterations = 20;
                    s.learningrate = 1.0;
                    break;
                case "iris":
                    s = new NNSettings();
                    s.layers = new int[]{2};
                    s.iterations = 500;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "iris.2d":
                    s = new NNSettings();
                    s.layers = new int[]{2};
                    s.iterations = 200;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "iris_test":
                    s = new NNSettings();
                    s.layers = new int[]{2};
                    s.iterations = 500;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "spiral":
                    s = new NNSettings();
                    s.layers = new int[]{72};
                    s.iterations = 8000;
                    s.use_regularization = false;
                    s.learningrate = 0.4;
                    break;
                case "gaussian":
                    s = new NNSettings();
                    s.layers = new int[]{8};
                    s.iterations = 50;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "flame":
                    s = new NNSettings();
                    s.layers = new int[]{16};
                    s.iterations = 1200;
                    s.use_regularization = true;
                    s.learningrate = 0.5;
                    break;
                case "jain":
                    s = new NNSettings();
                    s.layers = new int[]{16};
                    s.iterations = 300;
                    s.use_regularization = false;
                    s.learningrate = 0.8;
                    break;
                case "diabetes":
                    s = new NNSettings();
                    s.layers = new int[]{8};
                    s.iterations = 6000;
                    s.use_regularization = true;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "circle":
                    s = new NNSettings();
                    s.layers = new int[]{16};
                    s.iterations = 100;
                    s.use_regularization = false;
                    s.learningrate = 1.0;   
                    break;
                case "glass":
                    s = new NNSettings();
                    s.layers = new int[]{72};
                    s.iterations = 4000;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "mnist":
                    s = new NNSettings();
                    s.layers = new int[]{8};
                    s.iterations = 200;
                    s.use_regularization = false;
                    s.learningrate = 0.2;
                    s.normalization_type = Dataset.Norm_POS;
                    //Takes around 25 mins to train 200 iterations
                    break;
            }
        }
        if (type.equalsIgnoreCase("dnn"))
        {
            /**
             * Deep Neural Network classifiers 
             */
            switch (file) 
            {
                case "demo":
                    s = new NNSettings();
                    s.layers = new int[]{4,4};
                    s.iterations = 2000;
                    s.use_regularization = true;
                    s.learningrate = 0.1;
                    break;
                case "iris":
                    s = new NNSettings();
                    s.layers = new int[]{8,4};
                    s.iterations = 2000;
                    s.use_regularization = false;
                    s.learningrate = 0.8;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "iris_test":
                    s = new NNSettings();
                    s.layers = new int[]{8,4};
                    s.iterations = 2000;
                    s.use_regularization = false;
                    s.learningrate = 0.8;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "spiral":
                    s = new NNSettings();
                    s.layers = new int[]{42,24};
                    s.iterations = 8000;
                    s.use_regularization = false;
                    s.learningrate = 0.1;
                    break;
                case "diabetes":
                    s = new NNSettings();
                    s.layers = new int[]{24,12};
                    s.iterations = 8000;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "glass":
                    s = new NNSettings();
                    s.layers = new int[]{64,32}; //96.26
                    s.iterations = 6000;
                    s.use_regularization = false;
                    s.learningrate = 0.8;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
                case "circle":
                    s = new NNSettings();
                    s.layers = new int[]{12,8};
                    s.iterations = 100;
                    s.use_regularization = false;
                    s.learningrate = 1.0;
                    s.normalization_type = Dataset.Norm_NEGPOS;
                    break;
            }
        }
        
        return s;
    }
    
    /**
     * Returns the settings to use for kNN classifiers on the supplied datasets.
     * 
     * @param file Dataset identifier
     * @return Settings to use
     */
    public static KNNSettings getKNNSettings(String file)
    {
        KNNSettings s = null;
        
        switch(file)
        {
            case "demo":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "iris":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "iris.2d":
                s = new KNNSettings();
                s.K = 3;
                s.normalization_type = Dataset.Norm_NEGPOS;
                break;
            case "iris_test":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "spiral":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "diabetes":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "circle":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "glass":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "gaussian":
                s = new KNNSettings();
                s.K = 3;
                s.normalization_type = Dataset.Norm_NEGPOS;
                break;
            case "flame":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "jain":
                s = new KNNSettings();
                s.K = 3;
                break;
            case "mnist":
                s = new KNNSettings();
                s.K = 3;
                s.normalization_type = Dataset.Norm_POS;
                break;
        }
        
        return s;
    }
}
