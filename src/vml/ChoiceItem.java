
package vml;

/**
 * Container class needed to populate the experiments choice box in the GUI.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
class ChoiceItem
{
    /** Experiment id */
    protected String id;
    /** Classifier type */
    protected String type;
    /** Training data file */
    protected String file;

    /**
     * Creates a new choice item.
     * 
     * @param id Experiment id
     * @param type Classifier type
     * @param file Training data file
     */
    public ChoiceItem(String id, String type, String file)
    {
        this.id = id;
        this.type = type;
        this.file = file;
    }

    @Override
    public String toString()
    {
        return file + " (" + type + ")";
    }
}