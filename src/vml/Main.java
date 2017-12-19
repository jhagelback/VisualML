
package vml;

import java.io.File;

/**
 * Main class for the Visual ML application.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class Main 
{
    //Used to see if the specified experiment was found or not
    private static boolean found;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) 
    {
        args = new String[3];
        //args[0] = "-gen";
        //args[0] = "-exp";
        args[0] = "-gui";
        args[1] = "linear";
        args[2] = "iris";
        
        if (args[0].equalsIgnoreCase("-gui"))
        {
            runGUI();
        }
        else if (args[0].equalsIgnoreCase("-gen") || args[0].equalsIgnoreCase("-generator"))
        {
            runGenerator();
        }
        else if (args[0].equalsIgnoreCase("-exp") || args[0].equalsIgnoreCase("-experiment"))
        {
            if (args.length == 3)
            {
                runExperiment(args[1], args[2]);
                
                if (!found) System.err.println("Unable to find experiment " + args[1] + " " + args[2]);
            }
            else
            {
                System.err.println("Wrong arguments: -experiment [type] [dataset]");
            }
        }
        else
        {
            System.err.println("Wrong arguments: [-experiment|-gui|-generator] [args]");
        }
    }
    
    /**
     * Starts the GUI.
     */
    public static void runGUI()
    {
        new GUI();
    }
    
    /**
     * Runs the dataset generator.
     */
    public static void runGenerator()
    {
        //DataGenerator.circle();
        DataGenerator.fix();
    }
 
    /**
     * Runs an experiment.
     * 
     * @param type Type (knn, linear, nn, dnn)
     * @param file Dataset file
     */
    public static void runExperiment(String type, String file)
    {
        //Unset found
        found = false;
        
        if (type.equalsIgnoreCase("knn"))
        {
            /**
             * k-Nearest Neighbor classifiers
             */
            if (file.equalsIgnoreCase("iris")) evaluate(new KNN(3, 3), "iris", Dataset.Norm_NONE); // % 96.00%
            if (file.equalsIgnoreCase("iris_test")) evaluate(new KNN(3, 3), "iris_training", "iris_test", Dataset.Norm_NEGPOS); // 96.67%  96.67%
            if (file.equalsIgnoreCase("spiral")) evaluate(new KNN(3, 3), "spiral", Dataset.Norm_NONE); // % 99.33%
            if (file.equalsIgnoreCase("diabetes")) evaluate(new KNN(3, 2), "diabetes", Dataset.Norm_NONE); // 85.94%
            if (file.equalsIgnoreCase("circle")) evaluate(new KNN(3, 2), "circle", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("glass")) evaluate(new KNN(3, 7), "glass", Dataset.Norm_NONE); // 86.45%
            //Takes around 18 minutes for test set evaluation...
            if (file.equalsIgnoreCase("mnist")) evaluate(new KNN(3, 10), "mnist_train", "mnist_test", Dataset.Norm_POS); // 97.17% (test set, L2 dist)
            //if (file.equalsIgnoreCase("mnist")) evaluate(new KNN(3, 10), "mnist_train", "mnist_test", Dataset.Norm_POS); // 96.40% (test set, L1 dist)
        }
        
        if (type.equalsIgnoreCase("linear"))
        {
            /**
             * Linear classifiers 
             */
            if (file.equalsIgnoreCase("demo")) evaluate(new Linear(2, 3, 20, 0.1), "datademo", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("demo_fixed")) evaluate(new Linear(), "datademo", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("iris")) evaluate(new Linear(4, 3, 300, 0.1), "iris", Dataset.Norm_NONE); // 98%
            if (file.equalsIgnoreCase("iris_test")) evaluate(new Linear(4, 3, 300, 0.1), "iris_training", "iris_test", Dataset.Norm_NONE); // 97.50%  93.33%
            if (file.equalsIgnoreCase("spiral")) evaluate(new Linear(2, 3, 200, 0.1), "spiral", Dataset.Norm_NONE); // 49%
            if (file.equalsIgnoreCase("diabetes")) evaluate(new Linear(8, 2, 50, 0.8), "diabetes", Dataset.Norm_NEGPOS); // 77.21%
            if (file.equalsIgnoreCase("circle")) evaluate(new Linear(2, 2, 200, 0.1), "circle", Dataset.Norm_NONE); // 68.60%
            if (file.equalsIgnoreCase("glass")) evaluate(new Linear(9, 7, 200, 0.5), "glass", Dataset.Norm_NEGPOS); // 58.88%
            if (file.equalsIgnoreCase("mnist")) evaluate(new Linear(784, 10, 50, 0.1), "mnist_train", "mnist_test", Dataset.Norm_POS); // 87.58%  88.27%
            //if (file.equalsIgnoreCase("mnist")) evaluate(new Linear(784, 10, 1000, 0.4), "mnist_train", "mnist_test", Dataset.Norm_POS); // %  91.37%
        }
        
        if(type.equalsIgnoreCase("nn"))
        {
            /**
             * Neural Network classifiers 
             */
            if (file.equalsIgnoreCase("demo")) evaluate(new NN(2, 3, 8, 100, 0.5), "datademo", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("iris")) evaluate(new NN(4, 3, 4, 1000, 0.5), "iris", Dataset.Norm_NEGPOS); // 98.67% (98.6667% in Weka)
            if (file.equalsIgnoreCase("iris_test")) evaluate(new NN(4, 3, 4, 1000, 0.5), "iris_training", "iris_test", Dataset.Norm_NEGPOS); // 99.17%  96.67%
            if (file.equalsIgnoreCase("spiral")) evaluate(new NN(2, 3, 72, 8000, 0.4), "spiral", Dataset.Norm_NONE); // 99.33%
            if (file.equalsIgnoreCase("diabetes")) evaluate(new NN(8, 2, 8, 6000, 0.3), "diabetes", Dataset.Norm_NEGPOS); // 80.73% (80.599% in Weka)
            if (file.equalsIgnoreCase("circle")) evaluate(new NN(2, 2, 72, 1000, 0.4), "circle", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("glass")) evaluate(new NN(9, 7, 72, 9000, 0.3), "glass", Dataset.Norm_NEGPOS); // 86.92% (85.98% in Weka)
            
            if (file.equalsIgnoreCase("mnist")) evaluate(new NN(784, 10, 8, 200, 0.1), "mnist_train", "mnist_test", Dataset.Norm_POS); // 89.51%  89.58%
            //if (file.equalsIgnoreCase("mnist")) evaluate(new NN(784, 10, 12, 500, 0.05), "mnist_train", "mnist_test", Dataset.Norm_POS); // 88.06%  88.47%
        }
        
        if (type.equalsIgnoreCase("dnn"))
        {
            /**
             * Deep Neural Network classifiers 
             */
            if (file.equalsIgnoreCase("demo")) evaluate(new DeepNN(2, 3, 4, 4, 2000, 0.2), "datademo", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("iris")) evaluate(new DeepNN(4, 3, 8, 4, 2000, 0.2), "iris", Dataset.Norm_NEGPOS); // 98.67%
            if (file.equalsIgnoreCase("iris_test")) evaluate(new DeepNN(4, 3, 8, 4, 2000, 0.2), "iris_training", "iris_test", Dataset.Norm_NEGPOS); // 99.17%  96.67%
            if (file.equalsIgnoreCase("spiral")) evaluate(new DeepNN(2, 3, 42, 24, 12000, 0.08), "spiral", Dataset.Norm_NONE); // 99.33%
            if (file.equalsIgnoreCase("diabetes")) evaluate(new DeepNN(8, 2, 24, 8, 6000, 0.2), "diabetes", Dataset.Norm_NEGPOS); // 80.73%
            if (file.equalsIgnoreCase("circle")) evaluate(new DeepNN(2, 2, 12, 8, 1000, 0.1), "circle", Dataset.Norm_NEGPOS); // 100%
        }
    }
    
    /**
     * Trains and evaluates a classifier on a dataset.
     * 
     * @param c The classifier
     * @param dataset_name Train and test dataset
     * @param norm_type Normalization type (None, Pos, NegPos)
     */
    private static void evaluate(Classifier c, String dataset_name, int norm_type)
    {
        evaluate(c, dataset_name, null, norm_type);
    }
    
    /**
     * Trains and evaluates a classifier on a dataset.
     * 
     * @param c The classifier
     * @param dataset_name Train dataset
     * @param testset_name Test dataset
     * @param norm_type Normalization type (None, Pos, NegPos)
     */
    private static void evaluate(Classifier c, String dataset_name, String testset_name, int norm_type)
    {
        //Check if dataset is found
        File f = new File("data/" + dataset_name + ".csv");
        if (!f.exists()) return;
        
        //Set found
        found = true;
        
        //Read data
        DataSource reader = new DataSource("data/" + dataset_name + ".csv");
        Dataset data = reader.read();
        data.normalizeAttributes(norm_type);
        c.setData(data);
        
        //Train classifier
        long st = System.currentTimeMillis();
        c.train();
        long el = System.currentTimeMillis() - st;
        System.out.println("Training time: " + el + " ms");
        //Evaluate accuracy
        System.out.println("Performance (whole dataset):");
        c.evaluate(data);
        
        //Evaluate on test dataset (if it is specified)
        if (testset_name != null)
        {
            //Read test data
            reader = new DataSource("data/" + testset_name + ".csv");
            data = reader.read();
            data.normalizeAttributes(norm_type);
            
            //Evaluate accuracy
            System.out.println("Performance (test dataset):");
            c.evaluate(data);
        }
    }
}
