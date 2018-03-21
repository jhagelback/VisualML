
package vml;

import javafx.scene.paint.*;
import javafx.scene.canvas.*;

/**
 * Canvas for drawing decision boundaries and training data.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class VizCanvas extends Canvas
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
    private final int cell_w = 7;
    //True of panels is rendering, false otherwise
    public boolean updating;
    //Predicted class values
    private int[][] frame;
    
    /**
     * Inits a new renderpanel.
     */
    public VizCanvas()
    {
        super(707, 707);
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
        if (min < 0) shift = -min;
        
        //Generate frame
        build_frame();
    }
    
    /**
     * Updates the canvas.
     */
    public void update()
    {
        draw(this.getGraphicsContext2D());
    }
    
    /**
     * Draw canvas.
     * 
     * @param gc Graphics context
     */
    public void draw(GraphicsContext gc)
    {
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, getWidth(), getHeight());
        
        //Error check
        if (c == null) return;
        if (frame == null) return;
        
        //We are rendering
        updating = true;
        
        //Render cells
        for (int x = 0; x < 101; x++)
        {
            for (int y = 0; y < 101; y++)
            {
                //Find predicted class
                int label = frame[x][y];
                //Set color based on predicted class
                if (label == 0) gc.setFill(Color.rgb(255,219,194));
                if (label == 1) gc.setFill(Color.rgb(194,219,255));
                if (label == 2) gc.setFill(Color.rgb(219,255,194));
                //Render cell
                gc.fillRect(x * cell_w, y * cell_w, cell_w, cell_w);
            }
        }
        
        //Render known labels
        for (int i = 0; i < data.size(); i++)
        {
            //Iterate over each instance
            Instance inst = data.get(i);
            //Calculate cell (x,y) values
            //double x = (inst.x.get(0) / scale + shift) * 100.0;
            //double y = (inst.x.get(1) / scale + shift) * 100.0;
            double x = ( (inst.x.get(0)+shift) / scale) * 100.0;
            double y = ( (inst.x.get(1)+shift) / scale) * 100.0;
            
            //System.out.println("(" + inst.x.get(0) + "," + inst.x.get(1) + ") -> (" + x + "," + y + ")");
            
            //Draw outer border
            gc.setFill(Color.BLACK);
            gc.fillOval(x * cell_w, y * cell_w, cell_w, cell_w);
            //Set color based on the label
            if (inst.label == 0) gc.setFill(Color.rgb(204,102,0));
            if (inst.label == 1) gc.setFill(Color.rgb(0,102,204));
            if (inst.label == 2) gc.setFill(Color.rgb(102,204,0));
            //Render
            gc.fillOval(x * cell_w + 1, y * cell_w + 1, cell_w - 2, cell_w - 2);
        }
                
        //Done updating
        updating = false;
    }
    
    /**
     * Builds a new prediction frame to be rendered.
     */
    public void build_frame()
    {
        try
        {
            int[][] tmp = new int[101][101];
            
            //Calculate cell values
            for (int x = 0; x < 101; x++)
            {
                for (int y = 0; y < 101; y++)
                {
                    //Calculate instance values for this cell
                    double[] vals = new double[2];
                    //vals[0] = (x / 100.0 - shift) * scale;
                    //vals[1] = (y / 100.0 - shift) * scale;
                    vals[0] = (x / 100.0) * scale - shift;
                    vals[1] = (y / 100.0) * scale - shift;
                    //Create dataset for the instance
                    Dataset d = new Dataset();
                    Instance inst = new Instance(vals, 0);
                    d.add(inst);
                    //Classify the instance
                    c.activation(d);
                    int label = c.classify(0);
                    //Set label
                    tmp[x][y] = label;
                }
            }
            
            //If no errors occured, set reference to new frame
            frame = tmp;
        }
        catch (Exception ex)
        {
            
        }
    }
}
