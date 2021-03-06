
package vml;

import java.util.ArrayList;
import javafx.scene.paint.*;
import javafx.scene.canvas.*;

/**
 * Canvas for drawing decision boundaries and training data.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
class VizCanvas extends Canvas
{
    //The classifier
    private Classifier c;
    //Training dataset
    private Dataset data;
    //Data attribute scales
    private double scaleX;
    private double scaleY;
    //Data attribute shift
    private double shiftX;
    private double shiftY;
    //Width of cells in the rendered panel
    private final int cell_w = 7;
    //True of panels is rendering, false otherwise
    public boolean updating;
    //Predicted class values
    private int[][] frame;
    //Scale and shift settings
    private ArrayList<VizDatasetSettings> settings;
    
    /**
     * Inits a new renderpanel.
     */
    public VizCanvas()
    {
        super(700, 700);
        //Read settings
        settings = VizSetup.findDatasetSettings();
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
        
        //Init scale and shift settings
        init_settings();
        
        //Generate frame
        build_frame();
    }
    
    /**
     * Init scale and shift settings for this dataset.
     */
    private void init_settings()
    {
        //Default values
        scaleX = 1;
        scaleY = 1;
        shiftX = 0;
        shiftY = 0;
        
        //Check if we have custom settings for this dataset
        for (VizDatasetSettings st : settings)
        {
            if(data.getName().contains(st.dataset_file))
            {
                //Set custom settings
                scaleX = st.scale_x;
                scaleY = st.scale_y;
                shiftX = st.shift_x;
                shiftY = st.shift_y;
                return;
            }
        }
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
        //Background fill
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, getWidth(), getHeight());
        
        //Error check
        if (c == null) return;
        if (data.noInputs() != 2) 
        {
            gc.setFill(Color.RED);
            gc.fillText("Unable to visualize data", 150, 200);
            gc.fillText("Dataset has " + data.noInputs() + " attributes, while 2 is required", 150, 220);
            return;
        }
        if (frame == null) return;
        
        //We are rendering
        updating = true;
        
        //Render cells
        for (int x = 0; x < 100; x++)
        {
            for (int y = 0; y < 100; y++)
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
            double x = ((inst.x.get(0)-shiftX))/scaleX * 100.0;
            double y = ((inst.x.get(1)-shiftY))/scaleY * 100.0;
            
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
        //Error check
        if (data.noInputs() != 2) return;
        
        try
        {
            //Container for the test instances
            Dataset d = data.clone_empty();
            //Temporary frame
            int[][] tmp = new int[100][100];
            
            //Calculate cell values
            for (int x = 0; x < 100; x++)
            {
                for (int y = 0; y < 100; y++)
                {
                    //Calculate instance values for this cell
                    double[] vals = new double[2];
                    vals[0] = x / 100.0 * scaleX + shiftX;
                    vals[1] = y / 100.0 * scaleY + shiftY;
                    
                    //Create dataset for the instance
                    Instance inst = new Instance(vals, 0);
                    d.data.clear();
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
