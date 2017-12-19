
package vml;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.text.DecimalFormat;

/**
 * GUI for visualizing decision borders for a classifier.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class GUI extends JFrame
{
    //Panel to render stuff on
    private RenderPanel p;
    //The classifier
    private Classifier c;
    //Training dataset
    private Dataset data;
    //Number of iterations per step for the classifier
    private int it_steps;
    //Current iteration
    private int iteration = 0;
    //Iterations label
    private JLabel label_it;
    //Loss label
    private JLabel label_loss;
    //Accuracy label
    private JLabel label_acc;
    //Thread fun status
    private boolean running = false;
    
    //Output formatting
    private DecimalFormat df3 = new DecimalFormat("0.000");
    private DecimalFormat df1 = new DecimalFormat("0.0");
    
    /**
     * Instantiates the GUI.
     */
    public GUI()
    {
        initComponents();
    }
    
    /**
     * Updates the render panel.
     * 
     * @param loss Current classifier loss
     */
    public void update(double loss)
    {
        //Current classifier accuracy
        c.setData(data);
        double acc = c.evaluate(data);
        //Update text labels
        label_it.setText("Iteration: " + iteration);
        label_loss.setText("Loss: " + df3.format(loss));
        label_acc.setText("Accuracy: " + df1.format(acc) + "%");
        //Repaint panel
        p.updateUI();
        p.repaint();
    }
    
    /**
     * Adds the lmenu to the frame.
     */
    private void addMenu()
    {
        //Init lmenu
        JMenuBar menubar = new JMenuBar();   
        JMenu lmenu = new JMenu("Linear");
        JMenu nmenu = new JMenu("NN");
        JMenu dmenu = new JMenu("DeepNN");
        JMenu kmenu = new JMenu("kNN");
        
        //Linear tasks
        JMenuItem mitem = new JMenuItem("Demo");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new Linear(2, 3, 20, 0.05), "datademo", Dataset.Norm_NONE, 1);
        });
        lmenu.add(mitem);
        
        mitem = new JMenuItem("Demo fixed");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new Linear(), "datademo", Dataset.Norm_NONE, 1);
        });
        lmenu.add(mitem);
        
        mitem = new JMenuItem("Spiral");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new Linear(2, 3, 200, 0.1), "spiral", Dataset.Norm_NONE, 1);
        });
        lmenu.add(mitem);
        
        mitem = new JMenuItem("Iris.2D");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new Linear(2, 3, 20, 0.4), "iris.2D", Dataset.Norm_NEGPOS, 5);
        });
        lmenu.add(mitem);
        
        mitem = new JMenuItem("Circle");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new Linear(2, 2, 20, 0.1), "circle", Dataset.Norm_NONE, 1);
        });
        lmenu.add(mitem);
        
        //NN tasks
        mitem = new JMenuItem("Spiral");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new NN(2, 3, 72, 8000, 0.2), "spiral", Dataset.Norm_NONE, 100);
        });
        nmenu.add(mitem);
        
        mitem = new JMenuItem("Circle");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new NN(2, 2, 72, 20, 0.4), "circle", Dataset.Norm_NONE, 20);
        });
        nmenu.add(mitem);
        
        mitem = new JMenuItem("Gaussian");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new NN(2, 2, 72, 20, 0.4), "gaussian", Dataset.Norm_NEGPOS, 20);
        });
        nmenu.add(mitem);
        
        mitem = new JMenuItem("Flame");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new NN(2, 2, 72, 20, 0.4), "flame", Dataset.Norm_NONE, 100);
        });
        nmenu.add(mitem);
        
        mitem = new JMenuItem("Jain's toy problem");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new NN(2, 2, 12, 20, 0.2), "jain", Dataset.Norm_NONE, 100);
        });
        nmenu.add(mitem);
        
        //Deep NN tasks
        mitem = new JMenuItem("Spiral");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new DeepNN(2, 3, 42, 24, 8000, 0.05), "spiral", Dataset.Norm_NONE, 200);
        });
        dmenu.add(mitem);
        
        mitem = new JMenuItem("Circle");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new DeepNN(2, 2, 12, 8, 20, 0.1), "circle", Dataset.Norm_NONE, 50);
        });
        dmenu.add(mitem);
        
        //kNN tasks
        mitem = new JMenuItem("Demo");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(3, 3), "datademo", Dataset.Norm_NONE, 1);
        });
        kmenu.add(mitem);
        
        mitem = new JMenuItem("Spiral");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(3, 3), "spiral", Dataset.Norm_NONE, 1);
        });
        kmenu.add(mitem);
        
        mitem = new JMenuItem("Iris.2D");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(3, 3), "iris.2D", Dataset.Norm_NONE, 1);
        });
        kmenu.add(mitem);
        
        mitem = new JMenuItem("Circle");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(2, 3), "circle", Dataset.Norm_NONE, 1);
        });
        kmenu.add(mitem);
        
        mitem = new JMenuItem("Gaussian");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(2, 3), "gaussian", Dataset.Norm_NEGPOS, 1);
        });
        kmenu.add(mitem);
        
        mitem = new JMenuItem("Flame");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(2, 3), "flame", Dataset.Norm_NONE, 1);
        });
        kmenu.add(mitem);
        
        mitem = new JMenuItem("Jain's toy problem");
        mitem.addActionListener((ActionEvent event) -> {
            initClassifier(new KNN(2, 3), "jain", Dataset.Norm_NONE, 1);
        });
        kmenu.add(mitem);
        
        //Add menus to frame
        menubar.add(lmenu);
        menubar.add(nmenu);
        menubar.add(dmenu);
        menubar.add(kmenu);
        setJMenuBar(menubar);
    }
    
    /**
     * Inits a new task to visualize.
     * 
     * @param c The classifier
     * @param dataset_name Name of dataset
     * @param norm_type Normalization type (None, Pos, NegPos)
     * @param it_steps Iterations to take each step
     */
    private void initClassifier(Classifier c, String dataset_name, int norm_type, int it_steps)
    {
        //Read data
        DataSource reader = new DataSource("data/" + dataset_name + ".csv");
        data = reader.read();
        //Normalize
        data.normalizeAttributes(norm_type);
        //Set data to classifier
        c.setData(data);
        //Init stuff
        this.c = c;
        this.it_steps = it_steps;
        this.iteration = 0;
        p.setTask(c, data);
        //Update panel
        update(0);
    }
    
    /**
     * Inits all GUI components.
     */
    private void initComponents()
    {
        p = new RenderPanel();
        
        //Right side buttons/labels panel
        JPanel bp = new JPanel();
        bp.setLayout(null);
        bp.setPreferredSize(new Dimension(150, 100 * p.cell_w));
        
        //Button for iterating the classifier
        JButton step = new JButton("Iterate");
        step.addActionListener((ActionEvent e) -> {
            //Error check
            if (c == null) return;
            
            double loss = 0;
            for (int i = 0; i < it_steps; i++)
            {
                iteration++;
                loss = c.iterate();
            }
            update(loss);
        });
        step.setBounds(5, 10, 80, 25);
        bp.add(step);
        //Button for running the iteration thread
        JButton run = new JButton("Run");
        run.addActionListener((ActionEvent e) -> {
            //Error check
            if (c == null) return;
            //Don't run for KNN classifier
            if (c instanceof KNN) return;
            
            if (!running) 
            {
                running = true;
                run.setText("Stop");
                startThread();
            }
            else
            {
                running = false;
                run.setText("Start");
            }
        });
        run.setBounds(5, 45, 80, 25);
        bp.add(run);
        //Iterations label
        label_it = new JLabel("Iteration: " + iteration);
        label_it.setBounds(5, 80, 140, 20);
        bp.add(label_it);
        //Accuracy label
        label_acc = new JLabel("Accuracy: ");
        label_acc.setBounds(5, 110, 140, 20);
        bp.add(label_acc);
        //Loss label
        label_loss = new JLabel("Loss: ");
        label_loss.setBounds(5, 140, 140, 20);
        bp.add(label_loss);
        
        //Frame settings
        setSize(100 * p.cell_w + 180, 100 * p.cell_w + 70);
        setTitle("Visual ML");
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        getContentPane().setLayout(new FlowLayout());
        getContentPane().add(p);
        getContentPane().add(bp);
        addMenu();
        
        //Show window
        setVisible(true);
    }
    
    /**
     * Starts the thread for iterating the classifier.
     */
    private void startThread()
    {
        Runnable runnable = () -> { 
            while (running)
            {
                //Start time
                long st = System.currentTimeMillis();
                
                //Iterate classifier
                double loss = 0;
                for (int i = 0; i < it_steps; i++)
                {
                    iteration++;
                    loss = c.iterate();
                }
                update(loss);
                //Make sure the training data is set
                c.setData(data);
                
                //Wait for update to finish
                sleep(10);
                while (p.updating) sleep(10);
                
                //Check how much time is left
                long el = System.currentTimeMillis() - st;
                int rest = Math.max(100, 400 - (int)el);
                
                //Sleep
                sleep(rest);
            }
        };
        
        //Start thread
        Thread t = new Thread(runnable);
        t.start();
    }
    
    /**
     * Sets the current thread to sleep for some time.
     * 
     * @param ms Milliseconds to sleep
     */
    private void sleep(int ms)
    {
        try
        {
            Thread.sleep(ms);
        }
        catch (Exception ex)
        {
            System.err.println("  Thread sleep interrupted");
        }
    }
}