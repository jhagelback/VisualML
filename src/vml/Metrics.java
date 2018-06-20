
package vml;

import java.text.DecimalFormat;

/**
 * Calculates the following performance metrics from evaluating a test dataset:<br>
 * - Correctly classified instances<br>
 * - Accuracy<br>
 * - Precision<br>
 * - Recall<br>
 * - F-score<br>
 * - Confusion matrix
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Metrics 
{
    // Confusion matrix
    private Tensor2D cm;
    // Correctly classified instances
    private int correct;
    // Accuracy
    private double accuracy;
    // Precision, Recall and F-score matrix
    private Tensor2D pr;
    // The test dataset
    private Dataset data;
    
    // Tensor2D index for Precision value
    private final int precision_i = 0;
    // Tensor2D index for Recall value
    private final int recall_i = 1;
    // Tensor2D index for F-score value
    private final int f_i = 2;
    
    // Output formatting
    private static DecimalFormat df3 = new DecimalFormat("0.000");
    
    /**
     * Creates a new metrics instance.
     * 
     * @param data The dataset
     */
    public Metrics(Dataset data)
    {
        correct = 0;
        accuracy = 0;
        this.data = data;
        
        cm = Tensor2D.zeros(data.noCategories(), data.noCategories());
    }
    
    /**
     * Computes all metrics using the specified classifier.
     * 
     * @param c The classifier
     */
    public void compute(Classifier c)
    {
        compute_cm(c);
        calc_accuracy();
        calc_metrics();
    }
    
    /**
     * Computes the Confusion matrix.
     * 
     * @param cl The classifier
     */
    private void compute_cm(Classifier cl)
    {
        //Calculate accuracy and Confusion Tensor2D
        correct = 0;
        cm = Tensor2D.zeros(data.noCategories(), data.noCategories());
        for (int i = 0; i < data.size(); i++)
        {
            //Accuracy
            int pred_class = cl.classify(i);
            if (pred_class == data.get(i).label)
            {
                correct++;
            }
            //Confusion Tensor2D
            cm.v[data.get(i).label][pred_class]++;
        }       
    }
    
    /**
     * Calculates Precision, Recall and F-score metrics.
     */
    private void calc_metrics()
    {
        // Tensor2D to hold values
        pr = Tensor2D.zeros(cm.rows() + 1, 3);
        for (int r = 0; r < cm.rows(); r++)
        {
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
            
            //Insert into matrix
            pr.v[r][precision_i] = precision;
            pr.v[r][recall_i] = recall;
            pr.v[r][f_i] = f;
            
            //Average scores
            pr.v[cm.rows()][precision_i] += precision;
            pr.v[cm.rows()][recall_i] += recall;
            pr.v[cm.rows()][f_i] += f;
        }
        //Average scores
        pr.v[cm.rows()][precision_i] /= cm.rows();
        pr.v[cm.rows()][recall_i] /= cm.rows();
        pr.v[cm.rows()][f_i] /= cm.rows();
    }
    
    /**
     * Calculates the accuracy.
     */
    private void calc_accuracy()
    {
        double n_ok = 0;
        double n_tot = 0;
        for (int r = 0; r < cm.rows(); r++)
        {
            for (int c = 0; c < cm.columns(); c++)
            {
                if (r == c) 
                {
                    n_ok += cm.v[r][c];
                }
                n_tot += cm.v[r][c];
            }
        }
        
        accuracy = (double)n_ok / (double)n_tot * 100.0;
    }
    
    /**
     * Returns the accuracy.
     * 
     * @return Accuracy
     */
    public double getAccuracy()
    {
        return accuracy;
    }
    
    /**
     * Returns number of correctly classified instances.
     * 
     * @return Correctly classified instances
     */
    public int getCorrectlyClassified()
    {
        return correct;
    }
    
    /**
     * Returns total number of instances in the test dataset.
     * 
     * @return Total number of instances
     */
    public int getTotalInstances()
    {
        return data.size();
    }
    
    /**
     * Returns the Precision for a category value.
     * 
     * @param r The category 
     * @return Precision
     */
    public double getPrecision(int r)
    {
        return pr.v[r][precision_i];
    }
    
    /**
     * Returns the Recall for a category value.
     * 
     * @param r The category 
     * @return Recall
     */
    public double getRecall(int r)
    {
        return pr.v[r][recall_i];
    }
    
    /**
     * Returns the F-score for a category value.
     * 
     * @param r The category 
     * @return F-score
     */
    public double getFscore(int r)
    {
        return pr.v[r][f_i];
    }
    
    /**
     * Returns the average Precision for all category values.
     * 
     * @return Average Precision
     */
    public double getAvgPrecision()
    {
        return pr.v[pr.rows() - 1][precision_i];
    }
    
    /**
     * Returns the average Recall for all category values.
     * 
     * @return Average Recall
     */
    public double getAvgRecall()
    {
        return pr.v[pr.rows() - 1][recall_i];
    }
    
    /**
     * Returns the average F-score for all category values.
     * 
     * @return Average F-score
     */
    public double getAvgFscore()
    {
        return pr.v[pr.rows() - 1][f_i];
    }
    
    /**
     * Appends another metrics instance to this metrics instance.
     * Used in cross-validation.
     * 
     * @param m The metrics instances to append
     */
    public void append(Metrics m)
    {
        this.cm.add(m.cm, 1);
        
        calc_metrics();
        calc_accuracy();
    }
    
    /**
     * Formats a confusion matrix to a nice output.
     * 
     * @param out Logger for log info
     */
    public void format_conf_matrix(Logger out)
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
        str += Logger.format_spaces("", catlen);
        for (int c = 0; c < cm.columns(); c++)
        {
            str += "  " + Logger.format_spaces("[" + c + "]", strlen);
        }
        str += "\n";
        
        //Iterate over all values in the matrix
        for (int r = 0; r < cm.rows(); r++)
        {
            //Index rows
            str += Logger.format_spaces("[" + r + "]", catlen);
            //Values for each label
            for (int c = 0; c < cm.columns(); c++)
            {
                str += "  " + Logger.format_spaces((int)cm.v[r][c] + "", strlen);
            }
            //Label string
            str += "  -> " + data.getCategoryLabel(r);
            
            str += "\n";
        }
        
        //Print formatted matrix
        out.appendText(str);
    }
    
    /**
     * Formats Precision, Recall and F-score to a nice output.
     * 
     * @param out Logger for log info
     */
    public void format_scores(Logger out)
    {
        String str = "Metrics by category:\n";
        
        //Length of category string
        int catlen = (data.noCategories() + "").length() + 2;
        if (catlen < 4) catlen = 4;
        
        //Score columns
        str += Logger.format_spaces("", catlen);
        str += "  Precision  Recall  F-Score\n";
        
        //Iterate over all values in the matrix
        //to calculate Precision and Recall
        double[] avg = new double[3];
        for (int r = 0; r < cm.rows(); r++)
        {
            //Score rows
            str += Logger.format_spaces("[" + r + "]", catlen);
            
            //Precision
            String p_str = df3.format(getPrecision(r)) + "";
            str += "  " + Logger.format_spaces(p_str, 9);
            //Recall
            String r_str = df3.format(getRecall(r)) + "";
            str += "  " + Logger.format_spaces(r_str, 6);
            //F-score
            String f_str = df3.format(getFscore(r)) + "";
            str += "  " + Logger.format_spaces(f_str, 7);
            
            str += "\n";
        }
        
        //Score rows
        str += Logger.format_spaces("Avg:", catlen);
        //Precision
        String p_str = df3.format(getAvgPrecision()) + "";
        str += "  " + Logger.format_spaces(p_str, 9);
        //Recall
        String r_str = df3.format(getAvgRecall()) + "";
        str += "  " + Logger.format_spaces(r_str, 6);
        //F-score
        String f_str = df3.format(getAvgFscore()) + "";
        str += "  " + Logger.format_spaces(f_str, 7);
        
        out.appendText(str);
    }
}
