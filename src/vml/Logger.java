
package vml;

import java.util.ArrayList;

/**
 * Handles logging to either console or GUI.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
class Logger 
{
    //Enable or disable logging
    private boolean enabled = true;
    //Sets if console or GUI shall be used
    private boolean console = true;
    //Queue for GUI logging
    protected ArrayList<String> queue;
    
    /**
     * Creates a new console logger.
     * 
     * @return Console logger
     */
    public static Logger getConsoleLogger()
    {
        return new Logger(true);
    }
    
    /**
     * Creates a new GUI logger.
     * 
     * @return GUI logger
     */
    public static Logger getGUILogger()
    {
        return new Logger(false);
    }
    
    /**
     * Private constructor. Use statid get methods to create a logger.
     * 
     * @param console True of console shall be used, false if GUI shall be used
     */
    private Logger(boolean console)
    {
        this.console = console;
        queue = new ArrayList<>();
    }
    
    /**
     * Enables the logger.
     */
    public void enable()
    {
        enabled = true;
    }
    
    /**
     * Disables the logger.
     */
    public void disable()
    {
        enabled = false;
    }
    
    /**
     * Appends text to the logger.
     * 
     * @param text Text
     */
    public void appendText(String text)
    {
        if (!enabled) return;
        
        if (!console)
        {
            queue.add(text + "\n");
        }
        else
        {
            System.out.println(text);
        }
    }
    
    /**
     * Appends error message to the logger.
     * 
     * @param text Text
     */
    public void appendError(String text)
    {
        if (!enabled) return;
        
        if (!console)
        {
            queue.add(text + "\n");
        }
        else
        {
            System.err.println(text);
        }
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
