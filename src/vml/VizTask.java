
package vml;

/**
 * Task for visualizing training and classification of a dataset. The tasks are shown
 * as menu items in the GUI window.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
class VizTask 
{
    //Experiment id (in the experiments.xml file)
    String experiment_id = "";
    //Speed (training iterations per step)
    int speed = 1;
    //Name of the task
    String name = "";
    //Menu to add the task to
    String menu = "";
    
    /**
     * Creates a new visualization task.
     * 
     * @param experiment_id Experiment id (in the experiments.xml file)
     * @param speed Speed (training iterations per step)
     * @param name Name of the task
     * @param menu Menu to add the task to
     */
    public VizTask(String experiment_id, int speed, String name, String menu)
    {
        this.experiment_id = experiment_id;
        this.speed = speed;
        this.name = name;
        this.menu = menu;
    }
    
    @Override
    public String toString()
    {
        return name;
    }
}
