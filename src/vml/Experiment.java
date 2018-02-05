
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
        type = "knn";
        file = "glass";
        
        //Set to lower case
        type = type.toLowerCase();
        file = file.toLowerCase();
        
        if (type.equals("knn"))
        {
            KNNSettings s = ClassifierFactory.getKNNSettings(file);
            
            /**
             * k-Nearest Neighbor classifiers
             */
            if (file.equals("iris")) evaluate(ClassifierFactory.createKNN("iris", null, s)); // % 96.00%
            else if (file.equals("iris_test")) evaluate(ClassifierFactory.createKNN("iris_training", "iris_test", s)); // 96.67%  96.67%
            else if (file.equals("spiral")) evaluate(ClassifierFactory.createKNN("spiral", null, s)); // % 99.33%
            else if (file.equals("diabetes")) evaluate(ClassifierFactory.createKNN("diabetes", null, s)); // 85.94%
            else if (file.equals("circle")) evaluate(ClassifierFactory.createKNN("circle", null, s)); // 100%
            else if (file.equals("glass")) evaluate(ClassifierFactory.createKNN("glass", null, s)); // 86.45%
            else if (file.equals("mnist")) evaluate(ClassifierFactory.createKNN("mnist_train", "mnist_test", s)); // 97.17% (test set, L2 dist)
            else
            {
                System.err.println("Unknown dataset '" + file + "' for classifier KNN");
                System.exit(0);
            }
        }
        
        else if (type.equals("linear"))
        {
            LSettings s = ClassifierFactory.getLSettings(file);
            
            /**
             * Linear classifiers 
             */
            if (file.equals("demo")) evaluate(ClassifierFactory.createLinear("datademo", null, s)); // 100%
            else if (file.equals("demo_fixed")) evaluate(ClassifierFactory.createLinearDemo()); // 100%
            else if (file.equals("iris")) evaluate(ClassifierFactory.createLinear("iris", null, s)); // 98.00%
            else if (file.equals("iris.2d")) evaluate(ClassifierFactory.createLinear("iris.2d", null, s)); // 96.00%
            else if (file.equals("iris_test")) evaluate(ClassifierFactory.createLinear("iris_training", "iris_test", s)); //97.50%  93.33%
            else if (file.equals("spiral")) evaluate(ClassifierFactory.createLinear("spiral", null, s)); // 49.00%
            else if (file.equals("diabetes")) evaluate(ClassifierFactory.createLinear("diabetes", null, s)); // 77.21%
            else if (file.equals("circle")) evaluate(ClassifierFactory.createLinear("circle", null, s)); // 68.60%
            else if (file.equals("glass")) evaluate(ClassifierFactory.createLinear("glass", null, s)); // 58.88%
            //Takes around 30 mins to train
            else if (file.equals("mnist")) evaluate(ClassifierFactory.createLinear("mnist_train", "mnist_test", s));
            else
            {
                System.err.println("Unknown dataset '" + file + "' for classifier Linear");
                System.exit(0);
            }
        }
        
        else if(type.equals("nn"))
        {
            NNSettings s = ClassifierFactory.getNNSettings("nn", file);
            
            /**
             * Neural Network classifiers 
             */
            if (file.equals("demo")) evaluate(ClassifierFactory.createNN("datademo", null, s)); // 100%
            else if (file.equals("iris")) evaluate(ClassifierFactory.createNN("iris", null, s)); // 98.67%     
            else if (file.equals("iris.2d")) evaluate(ClassifierFactory.createNN("iris.2D", null, s)); // 96.00%    
            else if (file.equals("iris_test"))  evaluate(ClassifierFactory.createNN("iris_training", "iris_test", s)); // 99.17%  96.67%
            else if (file.equals("spiral")) evaluate(ClassifierFactory.createNN("spiral", null, s)); // 99.33%
            else if (file.equals("gaussian")) evaluate(ClassifierFactory.createNN("gaussian", null, s)); // 99.12%
            else if (file.equals("flame")) evaluate(ClassifierFactory.createNN("flame", null, s)); // 99.17%
            else if (file.equals("jain")) evaluate(ClassifierFactory.createNN("jain", null, s)); // 95.44%
            else if (file.equals("diabetes")) evaluate(ClassifierFactory.createNN("diabetes", null, s)); // 82.29%
            else if (file.equals("circle")) evaluate(ClassifierFactory.createNN("circle", null, s)); // 100%        
            else if (file.equals("glass")) evaluate(ClassifierFactory.createNN("glass", null, s)); // 90.65%     
            //Takes around 25 mins to train 200 iterations
            else if (file.equals("demo")) evaluate(ClassifierFactory.createNN("mnist_train", "mnist_test", s)); //88.48% 88.68% no momentum
            else
            {
                System.err.println("Unknown dataset '" + file + "' for classifier NN");
                System.exit(0);
            }
        }
        
        else if (type.equals("dnn"))
        {
            NNSettings s = ClassifierFactory.getNNSettings("dnn", file);
            
            /**
             * Deep Neural Network classifiers 
             */
            if (file.equals("demo")) evaluate(ClassifierFactory.createNN("datademo", null, s)); // 100%
            else if (file.equals("iris")) evaluate(ClassifierFactory.createNN("iris", null, s)); // 98.67%
            else if (file.equals("iris_test")) evaluate(ClassifierFactory.createNN("iris_training", "iris_test", s)); // 99.17%  96.67%
            else if (file.equals("spiral")) evaluate(ClassifierFactory.createNN("spiral", null, s)); // 99.33%
            else if (file.equals("diabetes")) evaluate(ClassifierFactory.createNN("diabetes", null, s)); // 92.45%
            else if (file.equals("glass")) evaluate(ClassifierFactory.createNN("glass", null, s)); // 96.26%
            else if (file.equals("circle")) evaluate(ClassifierFactory.createNN("circle", null, s)); // 100%
            else
            {
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
