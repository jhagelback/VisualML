
package vml;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import javax.swing.*;

/**
 * Renders the decision borders for a classifier and the known labels for
 * the training dataset.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class RenderPanel extends JPanel
{
    //The classifier
    private Classifier c;
    //Training dataset
    private Dataset data;
    //Data attribute scale
    private double scale;
    //Data attribute shift
    private double shift;
    //Width of cells in the rendered panel
    public int cell_w = 7;
    //True of panels is rendering, false otherwise
    public boolean updating;
    
    /**
     * Inits a new renderpanel.
     */
    public RenderPanel()
    {
        this.setPreferredSize(new Dimension(101 * cell_w, 101 * cell_w));
    }
    
    /**
     * Sets the current render task.
     * 
     * @param c The classifier
     * @param data Training dataset
     */
    public void setTask(Classifier c, Dataset data)
    {
        this.c = c;
        this.data = data;
        
        //Find max and min of both attributes
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < data.size(); i++)
        {
            Instance inst = data.get(i);
            
            double v1 = inst.x.get(0);
            if (v1 < min) min = v1;
            if (v1 > max) max = v1;
            
            double v2 = inst.x.get(1);
            if (v2 < min) min = v2;
            if (v2 > max) max = v2;
        }
        //Round to integers
        min = Math.floor(min);
        max = Math.ceil(max);
        
        //Scale factor of the data
        scale = max - min;
        //Shift factor of the data
        shift = 0;
        if (min < 0) shift = -min / 2.0;
        
        System.out.println("Using scale " + scale + " and shift " + shift);
    }
        
    @Override
    public void paint(Graphics gn) 
    {
        Graphics2D g = (Graphics2D)gn;
        
        //Fill background
        g.setColor(Color.white);
        g.fillRect(0, 0, this.getWidth(), this.getHeight());
        
        //Error check
        if (c == null) return;
        
        //We are rendering
        updating = true;
        
        //Render cells
        for (int x = 0; x < 101; x++)
        {
            for (int y = 0; y < 101; y++)
            {
                //Calculate instance values for this cell
                double[] vals = new double[2];
                vals[0] = (x / 100.0 - shift) * scale;
                vals[1] = (y / 100.0 - shift) * scale;
                //Create dataset for the instance
                Dataset d = new Dataset();
                Instance inst = new Instance(vals, 0);
                d.add(inst);
                //Classify the instance
                c.activation(d);
                int label = c.classify(0);
                //Set color based on predicted class
                if (label == 0) g.setColor(new Color(255,219,194));
                if (label == 1) g.setColor(new Color(194,219,255));
                if (label == 2) g.setColor(new Color(219,255,194));
                //Render cell
                g.fillRect(x * cell_w, y * cell_w, cell_w, cell_w);
            }
        }
        
        //Render known labels
        for (int i = 0; i < data.size(); i++)
        {
            //Iterate over each instance
            Instance inst = data.get(i);
            //Calculate cell (x,y) values
            double x = (inst.x.get(0) / scale + shift) * 100.0;
            double y = (inst.x.get(1) / scale + shift) * 100.0;
            //Draw outer border
            g.setColor(Color.black);
            g.fillRect((int)x*cell_w, (int)y*cell_w, cell_w, cell_w);
            //Set color based on the label
            if (inst.label == 0) g.setColor(new Color(204,102,0));
            if (inst.label == 1) g.setColor(new Color(0,102,204));
            if (inst.label == 2) g.setColor(new Color(102,204,0));
            //Render
            g.fillRect((int)x*cell_w+1, (int)y*cell_w+1, cell_w-2, cell_w-2);
        }
        
        //Done updating
        updating = false;
    }
}
