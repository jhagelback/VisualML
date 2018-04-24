
package vml;

import java.text.DecimalFormat;

/**
 * Used to run experiments without showing the GUI.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Experiment 
{
    //Output formatting
    private static DecimalFormat df = new DecimalFormat("0.00");
    
    /**
     * Runs an experiment. See the experiments.xml file for available
     * experiments.
     * 
     * @param id Experiment id
     * @param eval_train Sets if accuracy shall be evaluated on the training dataset
     * @param eval_test Sets if accuracy shall be evaluated on the test dataset
     * @param eval_cv Sets if accuracy shall be evaluated using 10-fold cross validation
     * @param out Logger for log info
     */
    public static void run(String id, boolean eval_train, boolean eval_test, boolean eval_cv, Logger out)
    {
        try
        {
            //Initialize classifier
            Classifier c = ClassifierFactory.build(id, out);
            
            //Train classifier
            if (eval_train || eval_test)
            {
                long st = System.currentTimeMillis();
                c.train(out);
                long el = System.currentTimeMillis() - st;
                out.appendText("Training time: " + Classifier.time_string(el));
            }

            //Evaluate accuracy on training and test datasets
            c.evaluate(eval_train, eval_test, out);
            //Evaluate accuracy using 10-fold cross validation
            if (eval_cv && c.data.size() >= 10)
            {
                run_cv(c, out);
            }
        }
        catch (Exception ex)
        {
            out.appendError("Unable to run experiment with id '" + id + "'");
        }
    }
    
    /**
     * Runs 10-fold cross validation.
     * 
     * @param c Classifier to use
     * @param out Logger for log info
     */
    private static void run_cv(Classifier c, Logger out)
    {
        out.appendText("\n10-fold Cross Validation");
        
        Dataset data = c.data;
        int size = data.size();
        int fold_size = size / 10;
        
        //Metrics
        Metrics m = new Metrics(data);
        
        long st = System.currentTimeMillis();
        //Iterate over all 10 folds
        for (int f = 0; f < 10; f++)
        {
            //Start and end index for this fold
            int start = f * fold_size;
            int end = (f + 1) * fold_size;
            if (f == 9 && end < size)
            {
                end = size;
            }
            
            //Create training and test sets
            Dataset train = data.getInverseSubset(start, end);
            Dataset test = data.getSubset(start, end);
            //Set datasets
            c.data = train;
            c.test = test;
            
            //Train classifier
            out.disable(); //Don't show log info
            c.train(out);
            out.enable();
            
            //Evaluate accuracy
            Metrics mt = c.test_accuracy();
            m.append(mt);
            
            out.appendText("    Fold " + (f + 1) + ": " + df.format(mt.getAccuracy()) + "%");
        }
        
        long el = System.currentTimeMillis() - st;
        out.appendText("Evaluation time: " + Classifier.time_string(el));
        out.appendText("Average accuracy: " + df.format(m.getAccuracy()) + "%");
        
        m.format_conf_matrix(out);
        m.format_scores(out);
    }
}
