
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
    //Output formatting
    private static DecimalFormat df3 = new DecimalFormat("0.000");
    
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
        //Create classifier
        Classifier c = ClassifierFactory.build(id);
        if (c == null)
        {
            out.appendError("Cannot find experiment with id '" + id + "'");
            return;
        }
        
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
        double[] acc = new double[10];
        double sum = 0;
        
        //Confusion matrix
        Matrix cm = Matrix.zeros(data.noCategories(), data.noCategories());
        
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
            acc[f] = c.test_accuracy();
            sum += acc[f];
            out.appendText("    Fold " + (f + 1) + ": " + df.format(acc[f]) + "%");
            
            cm.add(c.cm, 1.0);
        }
        sum /= 10.0;
        
        long el = System.currentTimeMillis() - st;
        out.appendText("Evaluation time: " + Classifier.time_string(el));
        out.appendText("Average accuracy: " + df.format(sum) + "%");
        
        Experiment.format_conf_matrix(cm, data, out);
        Experiment.format_scores(cm, data, out);
    }
    
    /**
     * Formats a confusion matrix to a nice output.
     * 
     * @param cm The confusion matrix
     * @param data The dataset
     * @param out Logger for log info
     */
    public static void format_conf_matrix(Matrix cm, Dataset data, Logger out)
    {
        //Length of value string
        int strlen = (data.size() + "").length();
        if (strlen < 3) strlen = 3;
        //Length of category string
        int catlen = (data.noCategories() + "").length() + 2;
        if (catlen < 3) catlen = 3;
        //Formatted string
        String str = "\nConfusion Matrix:\n";
        
        //Index columns
        str += format_spaces("", catlen);
        for (int c = 0; c < cm.columns(); c++)
        {
            str += "  " + format_spaces("[" + c + "]", strlen);
        }
        str += "\n";
        
        //Iterate over all values in the matrix
        for (int r = 0; r < cm.rows(); r++)
        {
            //Index rows
            str += format_spaces("[" + r + "]", catlen);
            //Values for each label
            for (int c = 0; c < cm.columns(); c++)
            {
                str += "  " + format_spaces((int)cm.v[r][c] + "", strlen);
            }
            //Label string
            str += "  -> " + data.getCategoryLabel(r);
            
            str += "\n";
        }
        
        //Print formatted matrix
        out.appendText(str);
    }
    
    /**
     * Calculates and outputs Precision, Recall and F-score.
     * 
     * @param cm The confusion matrix
     * @param data The dataset
     * @param out Logger for log info
     */
    public static void format_scores(Matrix cm, Dataset data, Logger out)
    {
        String str = "Metrics by category:\n";
        
        //Length of category string
        int catlen = (data.noCategories() + "").length() + 2;
        if (catlen < 4) catlen = 4;
        
        //Score columns
        str += format_spaces("", catlen);
        str += "  Precision  Recall  F-Score\n";
        
        //Iterate over all values in the matrix
        //to calculate Precision and Recall
        double[] avg = new double[3];
        for (int r = 0; r < cm.rows(); r++)
        {
            //Score rows
            str += format_spaces("[" + r + "]", catlen);
            
            //Calculate scores
            double tp = cm.v[r][r];
            double tpfp = cm.sum_row(r);
            double tpfn = cm.sum_col(r);
            double recall = 0;
            if (tpfn > 0) recall = tp / tpfn;
            double precision = 0;
            if (tpfp > 0) precision = tp / tpfp;
            double f = 0;
            if ((precision + recall) > 0) f = 2.0 * precision * recall / (precision + recall);
            //Average scores
            avg[0] += precision;
            avg[1] += recall;
            avg[2] += f;
            
            //Precision
            String p_str = df3.format(precision) + "";
            str += "  " + format_spaces(p_str, 9);
            //Recall
            String r_str = df3.format(recall) + "";
            str += "  " + format_spaces(r_str, 6);
            //F-score
            String f_str = df3.format(f) + "";
            str += "  " + format_spaces(f_str, 7);
            
            str += "\n";
        }
        //Calculate average
        avg[0] /= cm.rows();
        avg[1] /= cm.rows();
        avg[2] /= cm.rows();
        
        //Score rows
        str += format_spaces("Avg:", catlen);
        //Precision
        String p_str = df3.format(avg[0]) + "";
        str += "  " + format_spaces(p_str, 9);
        //Recall
        String r_str = df3.format(avg[1]) + "";
        str += "  " + format_spaces(r_str, 6);
        //F-score
        String f_str = df3.format(avg[2]) + "";
        str += "  " + format_spaces(f_str, 7);
        
        out.appendText(str);
    }
    
    /**
     * Formats a string to a specified length by putting whitespaces in front.
     * 
     * @param s The string
     * @param strlen Preferred length
     * @return Formatted string
     */
    protected static String format_spaces(String s, int strlen)
    {
        String str = "";
        int l = s.length();
        int rest = strlen - l;
        for (int i = 0; i < rest; i++)
        {
            str += " ";
        }
        return str + s;
    }
}
