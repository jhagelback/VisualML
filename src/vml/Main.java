
package vml;

import java.util.ArrayList;
import java.text.DecimalFormat;
import javafx.application.Application;
import javafx.event.*;
import javafx.geometry.*;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.animation.AnimationTimer;
import javafx.stage.WindowEvent;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;


/**
 * Main class for the VisualML application.
 * 
 * @author Johan Hagelbäck, Linnaeus University  (johan.hagelback@lnu.se)
 */
public class Main extends Application
{
    /**
     * Application version.
     */
    public static String version = "4.2";
    
    //Panel to render stuff on
    private VizCanvas p;
    //The classifier
    private Classifier c;
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
    //Run and Stop visualization buttons
    private Button bt1,bt2;
    //Experiments dropdown
    private ComboBox dd;
    //Evaluation checkboxes
    private CheckBox ecb1,ecb2,ecb3;
    //Run experiment button
    private Button rt;
    //Output textarea
    private TextArea output;
    //Output log info
    private Logger out;
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
        label_it.setPrefWidth(100);
        label_loss = new Label("Loss: ");
        label_loss.setPrefWidth(120);
        label_acc = new Label("Accuracy: ");
        label_acc.setPrefWidth(120);
        
        //Init buttons
        bt1 = new Button();
        bt1.setOnAction((ActionEvent e) -> {
            //Error check
            if (c == null) return;
            iterate();
        });
        bt1.setText("Iterate");
        bt1.setPrefWidth(90);
        bt2 = new Button();
        bt2.setOnAction((ActionEvent e) -> {
            //Error check
            if (c == null) return;
            //Don't run for non-iterable classifier
            if (!c.iterable_training())
            {
                return;
            }
            
            if (!running) 
            {
                running = true;
                bt1.setDisable(true);
                rt.setDisable(true);
                bt2.setText("Stop");
                
                //Thread run
                Runnable runnable = () -> { 
                    while (running)
                    {
                        iterate();
                        sleep(100);
                        
                        //Check if training is finished
                        if (c.training_done())
                        {
                            running = false;
                            return;
                        }
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
                bt1.setDisable(false);
                rt.setDisable(false);
            }
        });
        bt2.setText("Run");
        bt2.setPrefWidth(90);
        
        //Bottom panel (buttons and labels)
        HBox bp = new HBox();
        bp.setAlignment(Pos.CENTER);
        bp.setPrefSize(700, 60);
        bp.setPadding(new Insets(10));
        bp.setSpacing(10);
        bp.getChildren().addAll(bt1, bt2, label_it, label_loss, label_acc);
        bp.setPadding(new Insets(10));
        
        //Main layout panel
        GridPane pane = new GridPane(); 
        GridPane.setHalignment(bp, HPos.LEFT);
        GridPane.setValignment(bp, VPos.TOP);
        pane.setHgap(0);
        pane.setVgap(0);
        pane.setPadding(new Insets(10));
        
        //Right panel (experiments)
        Label lexp = new Label("Select Experiment");
        lexp.setTextFill(Color.web("#0076a3"));
        Label leval = new Label("Evaluation Options");
        leval.setTextFill(Color.web("#0076a3"));
        
        VBox rp = new VBox();
        rp.setAlignment(Pos.TOP_LEFT);
        rp.setPrefSize(370, 700);
        rp.setPadding(new Insets(10));
        rp.setSpacing(8);
        //Button
        rt = new Button();
        rt.setOnAction((ActionEvent e) -> {
            //Find selected experiment
            ChoiceItem sel = (ChoiceItem)dd.getValue();
            String sel_exp = sel.id;
            
            if (sel_exp != null && !running)
            {
                boolean eval_train = ecb1.isSelected();
                boolean eval_test = ecb2.isSelected();
                boolean eval_cv = ecb3.isSelected();
                output.setText("");
                new Thread() {
                    @Override
                    public void run() {
                        rt.setDisable(true);
                        bt1.setDisable(true);
                        bt2.setDisable(true);
                        running = true;
                        Experiment.run(sel_exp, eval_train, eval_test, eval_cv, out);
                        running = false;
                        rt.setDisable(false);
                        bt1.setDisable(false);
                        bt2.setDisable(false);
                    }
                }.start();
            }
        });
        rt.setText("Run Experiment");
        rt.setPrefWidth(120);
        //Experiments dropdown
        dd = new ComboBox();
        dd.setPrefWidth(220);
        //Fill experiments choice box
        ArrayList<ChoiceItem> exp = ClassifierFactory.findAvailableExperiments();
        dd.getItems().addAll(exp);
        //Evaluation checkbox
        ecb1 = new CheckBox("Training dataset");
        ecb1.setSelected(true);
        ecb2 = new CheckBox("Test dataset (if specified)");
        ecb2.setSelected(true);
        ecb3 = new CheckBox("Cross Validation");
        ecb3.setSelected(false);
        //Text field
        output = new TextArea("");
        output.setFont(Font.font("Menlo", FontWeight.NORMAL, 12));
        output.setEditable(false);
        out = Logger.getGUILogger();
        output.setPrefRowCount(29);
        rp.getChildren().addAll(lexp, dd, leval, ecb1, ecb2, ecb3, rt, output);
        
        out.appendText("\nWelcome to VisualML " + version);
        out.appendText("Developed by Johan Hagelbäck");
        out.appendText("Linnaeus University");
        out.appendText("johan.hagelback@lnu.se");
        
        //Add menu
        pane.add(buildMenu(), 0, 0);
        
        //Add right panel
        pane.add(rp, 1, 1);
        
        //Add bottom panel
        pane.add(bp, 0, 2);
        
        //Visualization canvas
        p = new VizCanvas();
        p.update();
        pane.add(p, 0, 1);
        
        //Create scene
        Scene scene = new Scene(pane, 1100, 810);
        
        //Start JavaFX stage
        primaryStage.setTitle("VisualML " + version);
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
                if (now - lastUpdate >= 200_000_000) {
                    p.update();
                    lastUpdate = now;
                    label_it.setText("Iteration: " + iteration);
                    label_loss.setText("Loss: " + df3.format(loss));
                    label_acc.setText("Accuracy: " + df1.format(acc) + "%");
                }
                if (!running)
                {
                    bt2.setText("Start");
                    bt1.setDisable(false);
                    rt.setDisable(false);
                }
            }
        }.start();
        
        //Timer for updating output textarea
        new AnimationTimer() {
            private long lastUpdate = 0 ;
            @Override
            public void handle(long now) {
                if (now - lastUpdate >= 10_000_000) {
                    lastUpdate = now;
                    while (!out.queue.isEmpty())
                    {
                        String text = out.queue.remove(0);
                        output.appendText(text);
                    }
                }
            }
        }.start();
    }
       
    /**
     * Adds the lmenu to the frame.
     */
    private MenuBar buildMenu()
    {
        //Init menu
        MenuBar menubar = new MenuBar();
        
        //Read tasks form visualization.xml file
        ArrayList<VizTask> tasks = VizSetup.findTasks();
        //Menus
        ArrayList<Menu> menus = new ArrayList<>();
        
        //Iterate through all tasks in the xml file
        for (VizTask task : tasks)
        {
            //Create menu item
            MenuItem mitem = new MenuItem(task.name);
            mitem.setOnAction((ActionEvent t) -> {
                initClassifier(ClassifierFactory.build(task.experiment_id, out), task.speed);
            });
            
            //Check if we already have the specified Menu
            Menu add_to = null;
            for (Menu m : menus)
            {
                if (m.getText().equalsIgnoreCase(task.menu))
                {
                    add_to = m;
                    break;
                }
            }
            //Menu not found. Create new.
            if (add_to == null)
            {
                add_to = new Menu(task.menu);
                menus.add(add_to);
            }
            
            //Add to list
            add_to.getItems().add(mitem);
        }
        
        //Add menus to GUI
        menubar.getMenus().addAll(menus);
        return menubar;
    }
    
    /**
     * Inits a new task to visualize.
     * 
     * @param c The classifier
     * @param it_steps Iterations to take each step
     */
    private void initClassifier(Classifier c, int it_steps)
    {
        //Special case for non-iterable classifiers
        if (!c.iterable_training())
        {
            c.iterate();
        }
        
        //Init stuff
        this.c = c;
        this.it_steps = it_steps;
        this.iteration = 0;
        this.loss = 0;
        this.acc = 0;
        p.setTask(c, c.getData());
        //Update panel
        p.update();
        
        Metrics m = c.train_accuracy();
        acc = m.getAccuracy();
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
            Metrics m = c.train_accuracy();
            acc = m.getAccuracy();
            
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
     * Runs the application.
     * Parameters:
     * -gui or no parameters: Runs the visualization GUI
     * -exp [id]: Runs the experiment with the specified id 
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args) 
    {
        //No args. Start GUI
        if (args.length == 0) 
        {
            args = new String[]{"-gui"};
        }
        
        if (args[0].equalsIgnoreCase("-gui"))
        {
            launch(args);
        }
        else if (args[0].equalsIgnoreCase("-exp"))
        {
            if (args.length >= 2)
            {
                boolean eval_train = true;
                boolean eval_test = true;
                boolean eval_cv = false;
                
                if (args.length == 3)
                {
                    if (args[2].contains("train")) eval_train = true;
                    else eval_train = false;
                    
                    if (args[2].contains("test")) eval_test = true;
                    else eval_test = false;
                    
                    if (args[2].contains("cv")) eval_cv = true;
                    else eval_cv = false;
                }
                
                Experiment.run(args[1], eval_train, eval_test, eval_cv, Logger.getConsoleLogger());
            }
            else
            {
                System.err.println("Wrong arguments: -exp [id] train|test|cv");
            }
            System.exit(0);
        }
        else if (args[0].equalsIgnoreCase("-dr"))
        {
            if (args.length >= 3)
            {
                String type = args[1];
                String filename = args[2];
                int vars = 1;
                if (args.length >= 4)
                {
                    vars = Integer.parseInt(args[3]);
                }
                //Run dimensionality reduction
                if (type.equalsIgnoreCase("PCA"))
                {
                    DimensionalityReduction dr = DimensionalityReduction.getPCA(filename, vars, Logger.getConsoleLogger());
                    dr.reduceAndSave();
                }
                if (type.equalsIgnoreCase("SVD"))
                {
                    DimensionalityReduction dr = DimensionalityReduction.getSVD(filename, vars, Logger.getConsoleLogger());
                    dr.reduceAndSave();
                }
            }
            System.exit(0);
        }
        else
        {
            System.err.println("Wrong arguments: [-exp|-gui][-dr] [args]");
            System.exit(1);
        }
    }      
}