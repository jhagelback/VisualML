
package vml;

import java.io.File;

/**
 * Used to run experiments without showing the GUI.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class Experiment 
{
    //Used to see if the specified experiment was found or not
    private static boolean found;
    
    /**
     * Runs an experiment.
     * 
     * @param type Type (knn, linear, nn, dnn)
     * @param file Dataset file
     * @return True if the specified experiment was found, false if not
     */
    public static boolean run(String type, String file)
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
            if (file.equalsIgnoreCase("diabetes")) evaluate(new Linear(8, 2, 50, 0.8), "diabetes", Dataset.Norm_NEGPOS); // 77.34%
            if (file.equalsIgnoreCase("circle")) evaluate(new Linear(2, 2, 200, 0.1), "circle", Dataset.Norm_NONE); // 68.60%
            if (file.equalsIgnoreCase("glass")) evaluate(new Linear(9, 7, 200, 0.5), "glass", Dataset.Norm_NEGPOS); // 58.41%
            //if (file.equalsIgnoreCase("mnist")) evaluate(new Linear2(784, 10, 50, 0.5), "mnist_train", "mnist_test", Dataset.Norm_POS); // 88.15%  88.89%
            if (file.equalsIgnoreCase("mnist")) evaluate(new Linear(784, 10, 10, 0.5), "mnist_train", "mnist_test", Dataset.Norm_POS); // 88.15%  88.89%
        }
        
        if(type.equalsIgnoreCase("nn"))
        {
            /**
             * Neural Network classifiers 
             */
            if (file.equalsIgnoreCase("demo")) evaluate(new NN(2, 3, 8, 100, 0.5), "datademo", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("iris")) evaluate(new NN(4, 3, 4, 1000, 0.5), "iris", Dataset.Norm_NEGPOS); // 98.67%
            if (file.equalsIgnoreCase("iris_test")) evaluate(new NN(4, 3, 4, 1000, 0.5), "iris_training", "iris_test", Dataset.Norm_NEGPOS); // 99.17%  96.67%
            if (file.equalsIgnoreCase("spiral")) evaluate(new NN(2, 3, 72, 8000, 0.4), "spiral", Dataset.Norm_NONE); // 99.33%
            if (file.equalsIgnoreCase("diabetes")) evaluate(new NN(8, 2, 8, 6000, 0.3), "diabetes", Dataset.Norm_NEGPOS); // 80.73% (80.599% in Weka)
            if (file.equalsIgnoreCase("circle")) evaluate(new NN(2, 2, 72, 1000, 0.4), "circle", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("glass")) evaluate(new NN(9, 7, 72, 9000, 0.3), "glass", Dataset.Norm_NEGPOS); // 85.98%
            if (file.equalsIgnoreCase("mnist")) evaluate(new NN(784, 10, 8, 1000, 0.05), "mnist_train", "mnist_test", Dataset.Norm_POS); // 89.25%  89.50%
        }
        
        if (type.equalsIgnoreCase("dnn"))
        {
            /**
             * Deep Neural Network classifiers 
             */
            if (file.equalsIgnoreCase("demo")) evaluate(new DeepNN(2, 3, 4, 4, 2000, 0.5), "datademo", Dataset.Norm_NONE); // 100%
            if (file.equalsIgnoreCase("iris")) evaluate(new DeepNN(4, 3, 8, 4, 2000, 0.3), "iris", Dataset.Norm_NEGPOS); // 98.67%
            if (file.equalsIgnoreCase("iris_test")) evaluate(new DeepNN(4, 3, 8, 4, 2000, 0.2), "iris_training", "iris_test", Dataset.Norm_NEGPOS); // 99.17%  96.67%
            if (file.equalsIgnoreCase("spiral")) evaluate(new DeepNN(2, 3, 42, 24, 12000, 0.08), "spiral", Dataset.Norm_NONE); // 99.33%
            if (file.equalsIgnoreCase("diabetes")) evaluate(new DeepNN(8, 2, 24, 12, 6000, 0.2), "diabetes", Dataset.Norm_NEGPOS); // 80.08%
            if (file.equalsIgnoreCase("circle")) evaluate(new DeepNN(2, 2, 12, 8, 1000, 0.1), "circle", Dataset.Norm_NEGPOS); // 100%
        }
        
        return found;
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
        st = System.currentTimeMillis();
        c.evaluate(data);
        el = System.currentTimeMillis() - st;
        System.out.println("Evaluation time: " + el + " ms");
        
        //Evaluate on test dataset (if it is specified)
        if (testset_name != null)
        {
            //Read test data
            reader = new DataSource("data/" + testset_name + ".csv");
            data = reader.read();
            data.normalizeAttributes(norm_type);
            
            //Evaluate accuracy
            System.out.println("Performance (test dataset):");
            st = System.currentTimeMillis();
            c.evaluate(data);
            el = System.currentTimeMillis() - st;
            System.out.println("Evaluation time: " + el + " ms");
        }
    }
}
