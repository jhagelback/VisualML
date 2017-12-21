
package vml;

import java.text.DecimalFormat;
import javafx.application.Application;
import javafx.event.*;
import javafx.geometry.*;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.animation.AnimationTimer;
import javafx.stage.WindowEvent;

/**
 * Main class for the Visual ML application.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class Main extends Application
{
    //Panel to render stuff on
    private VizCanvas p;
    //The classifier
    private Classifier c;
    //Training dataset
    private Dataset data;
    //Number of iterations per step for the classifier
    private int it_steps;
    //Current iteration
    private int iteration = 0;
    //Current loss
    private double loss = 0;
    //Current accuracy
    private double acc = 0;
    //Iterations label
    private Label label_it;
    //Loss label
    private Label label_loss;
    //Accuracy label
    private Label label_acc;
    //Thread run status
    private boolean running = false;
    
    //Output formatting
    private DecimalFormat df3 = new DecimalFormat("0.000");
    private DecimalFormat df1 = new DecimalFormat("0.0");
    
    /**
     * Start class for the JavaFX application.
     * 
     * @param primaryStage JavaFX stage
     */
    @Override
    public void start(Stage primaryStage) 
    {
        //Init labels
        label_it = new Label("Iteration: ");
        label_loss = new Label("Loss: ");
        label_acc = new Label("Accuracy: ");
        
        //Init buttons
        Button bt1 = new Button();
        bt1.setOnAction((ActionEvent e) -> {
            //Error check
            if (c == null) return;
            iterate();
        });
        bt1.setText("Iterate");
        bt1.setPrefWidth(90);
        Button bt2 = new Button();
        bt2.setOnAction((ActionEvent e) -> {
            //Error check
            if (c == null) return;
            //Don't run for KNN classifier
            if (c instanceof KNN) return;
            
            if (!running) 
            {
                running = true;
                bt2.setText("Stop");
                
                //Thread run
                Runnable runnable = () -> { 
                    while (running)
                    {
                        iterate();
                        sleep(100);
                    }
                };
                //Start thread
                Thread t = new Thread(runnable);
                t.start();
            }
            else
            {
                running = false;
                bt2.setText("Start");
            }
        });
        bt2.setText("Run");
        bt2.setPrefWidth(90);
        
        //Right panel (buttons and labels)
        VBox rp = new VBox();
        rp.setPrefSize(220, 100 * VizCanvas.cell_w);
        rp.setPadding(new Insets(10));
        rp.setSpacing(8);
        rp.getChildren().addAll(bt1, bt2, label_it, label_loss, label_acc);
        rp.setPadding(new Insets(10));
        
        //Main layout panel
        GridPane pane = new GridPane(); 
        GridPane.setHalignment(rp, HPos.LEFT);
        GridPane.setValignment(rp, VPos.TOP);
        pane.setHgap(0);
        pane.setVgap(0);
        pane.setPadding(new Insets(10));
        
        //Add menu
        pane.add(buildMenu(), 0, 0);
        
        //Add right panel
        pane.add(rp, 1, 1);
        
        //Visualization canvas
        p = new VizCanvas();
        p.update();
        pane.add(p, 0, 1);
        
        //Create scene
        Scene scene = new Scene(pane, 100 * VizCanvas.cell_w + 160, 100 * VizCanvas.cell_w + 55);
        
        //Start JavaFX stage
        primaryStage.setTitle("Visual ML");
        primaryStage.setScene(scene);
        primaryStage.show();
        
        //Close request
        primaryStage.setOnCloseRequest(null);
        primaryStage.setOnCloseRequest((WindowEvent we) -> {
            running = false;
            primaryStage.close();
        });
        
        //Timer for updating the visualization canvas
        new AnimationTimer() {
            private long lastUpdate = 0 ;
            @Override
            public void handle(long now) {
                if (now - lastUpdate >= 400_000_000) {
                    p.update();
                    lastUpdate = now;
                    label_it.setText("Iteration: " + iteration);
                    label_loss.setText("Loss: " + df3.format(loss));
                    label_acc.setText("Accuracy: " + df1.format(acc) + "%");
                }
            }
        }.start();
    }
    
    /**
     * Adds the lmenu to the frame.
     */
    private MenuBar buildMenu()
    {
        //Init lmenu
        MenuBar menubar = new MenuBar();   
        Menu lmenu = new Menu("Linear");
        Menu nmenu = new Menu("NN");
        Menu dmenu = new Menu("DeepNN");
        Menu kmenu = new Menu("kNN");
        
        //Linear tasks
        MenuItem mitem = new MenuItem("Demo");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new Linear(2, 3, 20, 0.05), "datademo", Dataset.Norm_NONE, 1);
        }); 
        lmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Demo fixed");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new Linear(), "datademo", Dataset.Norm_NONE, 1);
        }); 
        lmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Spiral");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new Linear(2, 3, 200, 0.1), "spiral", Dataset.Norm_NONE, 1);
        }); 
        lmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Iris.2D");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new Linear(2, 3, 20, 0.4), "iris.2D", Dataset.Norm_NEGPOS, 5);
        }); 
        lmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Circle");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new Linear(2, 2, 20, 0.1), "circle", Dataset.Norm_NONE, 1);
        }); 
        lmenu.getItems().add(mitem);
        
        //NN tasks
        mitem = new MenuItem("Spiral");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new NN(2, 3, 72, 8000, 0.2), "spiral", Dataset.Norm_NONE, 100);
        }); 
        nmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Circle");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new NN(2, 2, 72, 20, 0.4), "circle", Dataset.Norm_NONE, 20);
        }); 
        nmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Gaussian");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new NN(2, 2, 72, 20, 0.4), "gaussian", Dataset.Norm_NEGPOS, 20);
        }); 
        nmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Flame");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new NN(2, 2, 72, 20, 0.4), "flame", Dataset.Norm_NONE, 100);
        }); 
        nmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Jain's toy problem");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new NN(2, 2, 12, 20, 0.2), "jain", Dataset.Norm_NONE, 100);
        }); 
        nmenu.getItems().add(mitem);
        
        //Deep NN tasks
        mitem = new MenuItem("Spiral");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new DeepNN(2, 3, 42, 24, 8000, 0.05), "spiral", Dataset.Norm_NONE, 100);
        }); 
        dmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Circle");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new DeepNN(2, 2, 12, 8, 20, 0.1), "circle", Dataset.Norm_NONE, 50);
        }); 
        dmenu.getItems().add(mitem);
        
        //kNN tasks
        mitem = new MenuItem("Demo");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(3, 3), "datademo", Dataset.Norm_NONE, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Spiral");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(3, 3), "spiral", Dataset.Norm_NONE, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Iris.2D");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(3, 3), "iris.2D", Dataset.Norm_NONE, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Circle");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(2, 3), "circle", Dataset.Norm_NONE, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Gaussian");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(2, 3), "gaussian", Dataset.Norm_NEGPOS, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Flame");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(2, 3), "flame", Dataset.Norm_NONE, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        mitem = new MenuItem("Jain's toy problem");
        mitem.setOnAction((ActionEvent t) -> {
            initClassifier(new KNN(2, 3), "jain", Dataset.Norm_NONE, 1);
        }); 
        kmenu.getItems().add(mitem);
        
        //Add menus to frame
        menubar.getMenus().addAll(lmenu, nmenu, dmenu, kmenu);
        return menubar;
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
        this.loss = 0;
        this.acc = 0;
        p.setTask(c, data);
        //Update panel
        p.update();
        
        acc = c.evaluate(data);
        //Update data
        c.setData(data);
    }
    
    /**
     * Iterates the currenct classifier one step.
     */
    private void iterate()
    {
        try
        {
            //Iterate classifier
            for (int i = 0; i < it_steps; i++)
            {
                iteration++;
                loss = c.iterate();
            }
            //Current classifier accuracy
            c.setData(data);
            acc = c.evaluate(data);
            //Update data
            c.setData(data);
            
            //No errors occured, generate frame
            p.build_frame();
        }
        catch (Exception ex)
        {
            
        }
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
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) 
    {
        args = new String[3];
        //args[0] = "-gen";
        //args[0] = "-exp";
        args[0] = "-gui";
        args[1] = "linear";
        args[2] = "iris";
        
        if (args[0].equalsIgnoreCase("-gui"))
        {
            launch(args);
        }
        else if (args[0].equalsIgnoreCase("-gen") || args[0].equalsIgnoreCase("-generator"))
        {
            DataGenerator.run();
        }
        else if (args[0].equalsIgnoreCase("-exp") || args[0].equalsIgnoreCase("-experiment"))
        {
            if (args.length == 3)
            {
                boolean found = Experiment.run(args[1], args[2]);
                
                if (!found) System.err.println("Unable to find experiment " + args[1] + " " + args[2]);
            }
            else
            {
                System.err.println("Wrong arguments: -experiment [type] [dataset]");
            }
        }
        else
        {
            System.err.println("Wrong arguments: [-experiment|-gui|-generator] [args]");
        }
    }      
}