
package vml;

import java.util.*;

/**
 * CART (Classification And Regression Tree) tree classifier.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class CART extends Classifier
{
    //Internal test dataset
    private Dataset tdata;
     //Configuration settings
    private CARTSettings settings;
    //Possible class values
    private HashSet<Integer> classes;
    //Root node of the CART tree
    private Node root;
    
    /**
     * Internal class for tree nodes
     */
    protected class Node
    {
        // Subset for this node
        protected Dataset data;
        // Index of attribute to split at
        protected int a_index;
        // Value to split at
        protected double val;
        // Left branch
        protected Node left;
        // Right branch
        protected Node right;
        // Label (for leaf nodes)
        protected int label = -1;
        // Class distribution (for leaf nodes)
        protected Vector labels;
        
        /**
         * Creates a new node.
         * 
         * @param data Subset for this node
         * @param a_index Attribute to split it
         * @param val Value to split at
         */
        public Node(Dataset data, int a_index, double val)
        {
            this.data = data;
            this.a_index = a_index;
            this.val = val;
        }
        
        /**
         * Checks of this node is a terminal (leaf) node.
         * 
         * @return True if terminal node, false otherwise
         */
        public boolean is_terminal()
        {
            return (left == null || right == null);
        }
        
        /**
         * Calculates the predicted label and class distribution for a terminal node.
         */
        public void calc_label()
        {
            labels = Vector.zeros(data.noCategories());
            for (Instance i : data.data)
            {
                labels.v[i.label]++;
            }
            this.label = labels.argmax();
        }
    }
    
    /**
     * Creates a classifier.
     * 
     * @param data Training dataset
     * @param test Test dataset
     * @param settings Configuration settings for this classifier
     */
    public CART(Dataset data, Dataset test, CARTSettings settings)
    {
        //Set dataset
        this.data = data;
        this.test = test;
        
        //Size of dataset
        noCategories = data.noCategories();
        
        //Settings
        this.settings = settings;
        
        //Find all possible class values in the dataset
        classes = new HashSet<>();
        for (Instance inst : data.data)
        {
            if (!classes.contains(inst.label))
            {
                classes.add(inst.label);
            }
        }
    }
    
    /**
     * Trains the classifier.
     * 
     * @param o Logger for log info
     */
    @Override
    public void train(Logger o)
    {
        o.appendText("CART Classifier");
        o.appendText("Training data: " + data.getName());
        if (test != null)
        {
            o.appendText("Test data: " + test.getName());
        }
        
        iterate();
    }
    
    /**
     * Executes one training iteration.
     * 
     * @return Current loss
     */
    @Override
    public double iterate()
    {
        build_tree(settings.max_depth, settings.min_size, data);
        
        return 0;
    }
    
    /**
     * Performs activation for the specified dataset.
     * 
     * @param test Test dataset
     */
    @Override
    public void activation(Dataset test)
    {
        //Sets test dataset
        tdata = test;
    }
    
    /**
     * Classifies an instance in the dataset.
     * 
     * @param i Index of the instance
     * @return Predicted class value
     */
    @Override
    public int classify(int i)
    {
        return classify(root, tdata.get(i));
    }
    
    /**
     * Recursive classification.
     * 
     * @param node Current node
     * @param inst Instance to classify
     * @return Predicted label
     */
    public int classify(Node node, Instance inst)
    {
        int a_index = node.a_index;
        double val = node.val;
        
        //Check if left or right branch
        if (inst.x.get(a_index) < val)
        {
            if (!node.left.is_terminal())
            {
                //Not terminal node - keep iterating
                return classify(node.left, inst);
            }
            else
            {
                //Terminal node - return label
                return node.left.label;
            }
        }
        else
        {
            if (!node.right.is_terminal())
            {
                //Not terminal node - keep iterating
                return classify(node.right, inst);
            }
            else
            {
                //Terminal node - return label
                return node.right.label;
            }
        }
    }
    
    /**
     * Recursively builds the CART tree.
     * 
     * @param max_depth Max depth of the tree
     * @param min_size Min size of dataset for a split
     * @param data The dataset
     */
    private void build_tree(int max_depth, int min_size, Dataset data)
    {
        root = get_split(data);
        split(root, max_depth, min_size, 1);
    }
    
    /**
     * Tests to split a dataset at the specified attribute and value.
     * 
     * @param a_index Attribute index to split at
     * @param val Value to split at
     * @param data The dataset to split
     * @return Left and right branch after the split
     */
    private Node[] test_split(int a_index, double val, Dataset data)
    {
        Dataset left = data.clone_empty();
        Dataset right = data.clone_empty();
        
        //Iterate over the whole dataset and put instances in left or right branch
        for (Instance i : data.data)
        {
            if (i.x.get(a_index) < val) left.add(i);
            else right.add(i);
        }
        
        //Result branches
        Node[] groups = new Node[2];
        groups[0] = new Node(left, a_index, val);
        groups[1] = new Node(right, a_index, val);
        
        return groups;
    }
    
    /**
     * Counts the number of instances of the specified class label.
     * 
     * @param data The dataset
     * @param class_val The class label
     * @return Number of instances of the class
     */
    private double count(Dataset data, Integer class_val)
    {
        double cnt = 0;
        for (Instance i : data.data)
        {
            if (i.label == class_val)
            {
                cnt++;
            }
        }
        return cnt;
    }
    
    /**
     * Calculates the Gini index for two nodes.
     * 
     * @param groups Left and right branch nodes
     * @param classes Possible class label values
     * @return Gini index value
     */
    private double gini_index(Node[] groups, HashSet<Integer> classes)
    {
        //Total number of instances
        double n_instances = groups[0].data.size() + groups[1].data.size();
        //Gini index
        double gini = 0.0;
        
        //Iterate over both groups
        for (Node n : groups)
        {
            double size = n.data.size();
            double score = 0.0;
            
            //Calculate score
            if (size > 0)
            {
                for (Integer class_val : classes)
                {
                    double p = count(n.data, class_val) / size;
                    score += p * p;
                }
                
                //Update gini index
                gini += (1.0 - score) * (size / n_instances);
            }
        }
        
        return gini;
    }
    
    /**
     * Search for and splits the dataset at the best attribute-value combination.
     * 
     * @param data Dataset to split
     * @return Node the dataset splitted in a left and a right branch
     */
    public Node get_split(Dataset data)
    {
        //Init variables
        double b_index = -1;
        double b_value = 0;
        double b_score = Double.MAX_VALUE;
        Node[] b_groups = null; 
        
        //Iterate over all attributes...
        for (int a = 0; a < data.noInputs(); a++)
        {
            //... and instances
            for (Instance inst : data.data)
            {
                //Current attribute value
                double val = inst.x.get(a);
                
                //Test to split at this attribute-value combination
                Node[] groups = test_split(a, val, data);
                //Calculate Gini index for the split
                double gini = gini_index(groups, classes);
                //Check if we have a new best split
                if (gini < b_score)
                {
                    b_index = a;
                    b_value = val;
                    b_score = gini;
                    b_groups = groups;
                }
            }
        }
        
        //Create result node with the dataset splitted into a
        //left and right branch
        Node n = new Node(data, (int)b_index, b_value);
        n.left = b_groups[0];
        n.right = b_groups[1];
        
        return n;
    }
    
    /**
     * Recursive split of the dataset.
     * 
     * @param node Current node
     * @param max_depth Max depth of the tree
     * @param min_size Minimum size of dataset for a split
     * @param depth Current depth
     */
    private void split(Node node, int max_depth, int min_size, int depth)
    {
        //Left and right branch nodes
        Node left = node.left;
        Node right = node.right;

        //No split since left or right is null
        if (left == null || right == null)
        {
            //Terminal node - calculate label
            node.calc_label();
            return;
        }
        //Check for max depth
        if (depth >= max_depth)
        {
            //Terminal nodes - calculate labels
            node.left.calc_label();
            node.right.calc_label();
            return;
        }
        //Process left child
        if (left.data.size() <= min_size)
        {
            //Terminal node - calculate label
            node.left.calc_label();
        }
        else
        {
            node.left = get_split(left.data);
            //Recursive call
            split(node.left, max_depth, min_size, depth + 1);
        }
        //Process right child
        if (right.data.size() <= min_size)
        {
            //Terminal node - calculate label
            node.right.calc_label();
        }
        else
        {
            node.right = get_split(right.data);
            //Recursive call
            split(node.right, max_depth, min_size, depth + 1);
        }
    }
}
