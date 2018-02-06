
package vml;

/**
 * Used to run experiments without showing the GUI.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Experiment 
{
    /**
     * Runs an experiment. See the experiments.xml file for available
     * experiments.
     * 
     * @param id Experiment id
     */
    public static void run(String id)
    {
        Classifier c = ClassifierFactory.build(id);
        if (c == null)
        {
            System.err.println("Cannot find experiment with id '" + id + "'");
            System.exit(1);
        }
        
        //Train classifier
        long st = System.currentTimeMillis();
        c.train();
        long el = System.currentTimeMillis() - st;
        System.out.println("Training time: " + el + " ms");
        //Evaluate accuracy
        c.evaluate();
    }
}
