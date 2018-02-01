
package vml;

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
     */
    public static void run(String type, String file)
    {
        type = "linear";
        file = "mnist";
        
        //Set to lower case
        type = type.toLowerCase();
        file = file.toLowerCase();
        
        if (type.equals("knn"))
        {
            /**
             * k-Nearest Neighbor classifiers
             */
            switch (file) 
            {
                case "iris":
                    evaluate(ClassifierFactory.createKNN("iris", null, 3, Dataset.Norm_NONE)); // % 96.00% N
                    break;
                case "iris_test":
                    evaluate(ClassifierFactory.createKNN("iris_training", "iris_test", 3, Dataset.Norm_NONE)); // 96.67%  96.67% N
                    break;
                case "spiral":
                    evaluate(ClassifierFactory.createKNN("spiral", null, 3, Dataset.Norm_NONE)); // % 99.33% N
                    break;
                case "diabetes":
                    evaluate(ClassifierFactory.createKNN("diabetes", null, 3, Dataset.Norm_NONE)); // 85.94% N
                    break;
                case "circle":
                    evaluate(ClassifierFactory.createKNN("circle", null, 3, Dataset.Norm_NONE)); // 100% N
                    break;
                case "glass":
                    evaluate(ClassifierFactory.createKNN("glass", null, 3, Dataset.Norm_NONE)); // 86.45% N
                    break;
                case "mnist":
                    evaluate(ClassifierFactory.createKNN("mnist_train", "mnist_test", 3, Dataset.Norm_POS)); // 97.17% (test set, L2 dist)
                    break;
                default:
                    System.err.println("Unknown dataset '" + file + "' for classifier KNN");
                    System.exit(0);
            }
        }
        
        else if (type.equals("linear"))
        {
            /**
             * Linear classifiers 
             */
            switch (file) 
            {
                case "demo":
                    evaluate(ClassifierFactory.createLinear("datademo", null, 10, 1.0, Dataset.Norm_NONE)); // 100% N
                    break;
                case "demo_fixed":
                    evaluate(ClassifierFactory.createLinearDemo()); // 100% N
                    break;
                case "iris":
                    evaluate(ClassifierFactory.createLinear("iris", null, 300, 0.1, Dataset.Norm_NONE)); // 98% N
                    break;
                case "iris.2d":
                    evaluate(ClassifierFactory.createLinear("iris.2D", null, 50, 1.0, Dataset.Norm_NEGPOS)); // 96.00% N
                    break;
                case "iris_test":
                    evaluate(ClassifierFactory.createLinear("iris_training", "iris_test", 300, 0.1, Dataset.Norm_NONE)); // 97.50%  93.33% N
                    break;
                case "spiral":
                    evaluate(ClassifierFactory.createLinear("spiral", null, 200, 0.1, Dataset.Norm_NONE)); // 49% N
                    break;
                case "diabetes":
                    evaluate(ClassifierFactory.createLinear("diabetes", null, 40, 1.0, Dataset.Norm_NEGPOS)); // 77.21% N
                    break;
                case "circle":
                    evaluate(ClassifierFactory.createLinear("circle", null, 20, 1.0, Dataset.Norm_NONE)); // 68.60% N
                    break;
                case "glass":
                    evaluate(ClassifierFactory.createLinear("glass", null, 50, 1.0, Dataset.Norm_NEGPOS)); // 58.88% N
                    break;
                case "mnist":
                    //evaluate(ClassifierFactory.createLinear("mnist_train", "mnist_test", 50, 1.0, Dataset.Norm_POS)); // 88.86%  89.55%
                    //evaluate(ClassifierFactory.createLinear("mnist_train", "mnist_test", 100, 1.0, Dataset.Norm_POS)); // 89.99%  90.49%
                    evaluate(ClassifierFactory.createLinear("mnist_train", "mnist_test", 200, 1.0, Dataset.Norm_POS)); // 89.99%  90.49%
                    break;
                default:
                    System.err.println("Unknown dataset '" + file + "' for classifier Linear");
                    System.exit(0);
            }
        }
        
        else if(type.equals("nn"))
        {
            /**
             * Neural Network classifiers 
             */
            switch (file) 
            {
                case "demo":
                    evaluate(ClassifierFactory.createNN("datademo", null, 8, 20, 1.0, Dataset.Norm_NONE)); // 100% N
                    break;
                case "iris":
                    evaluate(ClassifierFactory.createNN("iris", null, 2, 500, 1.0, Dataset.Norm_NEGPOS)); // 98.67% N     
                    break;
                case "iris.2d":
                    evaluate(ClassifierFactory.createNN("iris.2D", null, 2, 200, 1.0, Dataset.Norm_NEGPOS)); // 96.00% N     
                    break;
                case "iris_test":
                    evaluate(ClassifierFactory.createNN("iris_training", "iris_test", 2, 500, 1.0, Dataset.Norm_NEGPOS)); // 99.17%  96.67% N
                    break;
                case "spiral":
                    evaluate(ClassifierFactory.createNN("spiral", null, 72, 8000, 0.4, Dataset.Norm_NONE)); // 99.33% N
                    break;
                case "gaussian":
                    evaluate(ClassifierFactory.createNN("gaussian", null, 8, 50, 1.0, Dataset.Norm_NEGPOS)); // 99.12% N
                    break;
                case "flame":
                    evaluate(ClassifierFactory.createNN("flame", null, 16, 1200, 0.5, Dataset.Norm_NONE)); // 99.17% N
                    break;
                case "jain":
                    evaluate(ClassifierFactory.createNN("jain", null, 16, 300, 0.8, Dataset.Norm_NONE)); // 95.71% N
                    break;
                case "diabetes":
                    evaluate(ClassifierFactory.createNN("diabetes", null, 8, 6000, 1.0, Dataset.Norm_NEGPOS)); // 81.12% N
                    break;
                case "circle":
                    evaluate(ClassifierFactory.createNN("circle", null, 16, 100, 1.0, Dataset.Norm_NONE)); // 100% N         
                    break;
                case "glass":
                    evaluate(ClassifierFactory.createNN("glass", null, 72, 2000, 1.0, Dataset.Norm_NEGPOS)); // 86.45% N     
                    break;
                case "mnist":
                    evaluate(ClassifierFactory.createNN("mnist_train", "mnist_test", 8, 100, 1.0, Dataset.Norm_POS)); // 89.25%  89.50%
                    break;
                default:
                    System.err.println("Unknown dataset '" + file + "' for classifier NN");
                    System.exit(0);
            }
        }
        
        else if (type.equals("dnn"))
        {
            /**
             * Deep Neural Network classifiers 
             */
            switch (file) 
            {
                case "demo":
                    evaluate(ClassifierFactory.createDNN("datademo", null, 4, 4, 2000, 0.1, Dataset.Norm_NONE)); // 100% N
                    break;
                case "iris":
                    evaluate(ClassifierFactory.createDNN("iris", null, 8, 4, 500, 1.0, Dataset.Norm_NEGPOS)); // 98.00% N
                    break;
                case "iris_test":
                    evaluate(ClassifierFactory.createDNN("iris_training", "iris_test", 8, 4, 500, 1.0, Dataset.Norm_NEGPOS)); // 98.33%  96.67% N
                    break;
                case "spiral":
                    evaluate(ClassifierFactory.createDNN("spiral", null, 42, 24, 8000, 0.1, Dataset.Norm_NONE)); // 99.33% N
                    break;
                case "diabetes":
                    evaluate(ClassifierFactory.createDNN("diabetes", null, 24, 12, 5000, 1.0, Dataset.Norm_NEGPOS)); // 81.64% N
                    break;
                case "circle":
                    evaluate(ClassifierFactory.createDNN("circle", null, 12, 8, 100, 1.0, Dataset.Norm_NEGPOS)); // 100% N
                    break;
                default:
                    System.err.println("Unknown dataset '" + file + "' for classifier DNN");
                    System.exit(0);
            }
        }
        
        else
        {
            System.err.println("Unknown classifier '" + type + "'");
            System.exit(0);
        }
    }
    
    /**
     * Trains a classifier on a training dataset and evaluates on the training dataset
     * and, if specified, test dataset.
     * 
     * @param c The classifier
     */
    private static void evaluate(Classifier c)
    {
        //Train classifier
        long st = System.currentTimeMillis();
        c.train();
        long el = System.currentTimeMillis() - st;
        System.out.println("Training time: " + el + " ms");
        //Evaluate accuracy
        c.evaluate();
    }
}
