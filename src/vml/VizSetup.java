
package vml;

import java.io.File;
import java.util.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Reads the visualization.xml file containing the datasets and classifiers to be
 * visualized in the GUI.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
class VizSetup 
{
    //visualization.xml root node
    private static Element root;
    
    /**
     * Returns a list of all visualization tasks in the visualization.xml file.
     * 
     * @return List of experiments
     */
    static ArrayList<VizTask> findTasks()
    {
        ArrayList<VizTask> tasks = new ArrayList<>();
        
        try
        {
            //Open xml file (if not already read)
            if (root == null) openXML();
            
            //Search for Task nodes
            NodeList nodes = root.getElementsByTagName("Task");
            for (int n = 0; n < nodes.getLength(); n++)
            {
                Node node = nodes.item(n);
                //Convert from Node to Element
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element e = (Element) node;
                    //Find task settings
                    String exp_id = e.getElementsByTagName("ExperimentId").item(0).getTextContent();
                    int speed = 0;
                    if (e.getElementsByTagName("Speed").getLength() != 0)
                    {
                        speed = Integer.parseInt(e.getElementsByTagName("Speed").item(0).getTextContent());
                    }
                    String name = e.getElementsByTagName("Name").item(0).getTextContent();
                    String menu = e.getElementsByTagName("Menu").item(0).getTextContent();
                    //Add to list
                    tasks.add(new VizTask(exp_id, speed, name, menu));
                }
            }
        }
        catch (Exception ex)
        {
            //Return what has been found so far
        }
        
        return tasks;
    }
    
    /**
     * Returns a list of all experiments in the experiments.xml file.
     * 
     * @return List of experiments
     */
    static ArrayList<VizDatasetSettings> findDatasetSettings()
    {
        ArrayList<VizDatasetSettings> settings = new ArrayList<>();
        
        try
        {
            //Open xml file (if not already read)
            if (root == null) openXML();
            
            //Search for experiment nodes
            NodeList nodes = root.getElementsByTagName("Dataset");
            for (int n = 0; n < nodes.getLength(); n++)
            {
                Node node = nodes.item(n);
                //Convert from Node to Element
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element e = (Element) node;
                    //Find dataset settings
                    String dataset_file = e.getElementsByTagName("DatasetFile").item(0).getTextContent();
                    double scale_x = 1;
                    double scale_y = 1;
                    double shift_x = 0;
                    double shift_y = 0;
                    if (e.getElementsByTagName("X-scale").getLength() != 0)
                    {
                        scale_x = Double.parseDouble(e.getElementsByTagName("X-scale").item(0).getTextContent());
                    }
                    if (e.getElementsByTagName("Y-scale").getLength() != 0)
                    {
                        scale_y = Double.parseDouble(e.getElementsByTagName("Y-scale").item(0).getTextContent());
                    }
                    if (e.getElementsByTagName("X-shift").getLength() != 0)
                    {
                        shift_x = Double.parseDouble(e.getElementsByTagName("X-shift").item(0).getTextContent());
                    }
                    if (e.getElementsByTagName("Y-shift").getLength() != 0)
                    {
                        shift_y = Double.parseDouble(e.getElementsByTagName("Y-shift").item(0).getTextContent());
                    }
                    //Add to list
                    settings.add(new VizDatasetSettings(dataset_file, scale_x, scale_y, shift_x, shift_y));
                }
            }
        }
        catch (Exception ex)
        {
            //Return what has been found so far
        }
        
        return settings;
    }
    
    /**
     * Reads the visualization.xml file into memory.
     * 
     * @throws java.lang.Exception If unable to open visualization xml file
     */
    private static void openXML() throws Exception
    {
        //Read xml file, if not already read into memory
        try
        {
            //Check if file exists and if it has been modified
            File xml = new File("visualization.xml");
            if (!xml.exists())
            {
                System.err.println("Visualization XML file does not exist");
                System.exit(1);
            }
            
            if (root == null)
            {
                DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
                DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
                Document doc = dBuilder.parse(xml);
                root = doc.getDocumentElement();
            }
        }
        catch (Exception ex)
        {
            throw ex;
        }
    }
}
