
package vml;

import java.io.*;
import java.text.DecimalFormat;
import java.util.Random;

/**
 * Used to generate some datasets.
 * 
 * @author Johan Hagelbäck (johan.hagelback@gmail.com)
 */
public class DataGenerator 
{
    /**
     * Runs the dataset generator.
     */
    public static void run()
    {
        circle();
        //fix();
    } 
    
    /**
     * Generates a dataset with an outer non-filled circle of one class, and a smaller
     * inner filled circle of the other class. Useful for showing the shortcomings of
     * linear classifiers.
     */
    public static void circle()
    {
        System.out.print("Generating circle dataset ... ");
        try
        {
            Random rnd = new Random(2);
            
            BufferedWriter o = new BufferedWriter(new FileWriter("data/circle.csv"));
            o.write("x,y,class\n");
            DecimalFormat df = new DecimalFormat("0.00"); 
            
            for (int x = -10; x <= 10; x++)
            {
                for (int y = -10; y <= 10; y++)
                {
                    //Check if we are in the outer or inner circle
                    int cat = 1;
                    double dist = Math.sqrt( Math.pow(x-0,2) + Math.pow(y-0, 2));
                    if (dist <= 5) cat = 0;

                    //Convert to decimal values, and add some random noise
                    double xv = (double)x/10.0 + (rnd.nextDouble()-0.5)*0.07;
                    double yv = (double)y/10.0 + (rnd.nextDouble()-0.5)*0.07;
                    
                    //Check if this point shall be added or not
                    //Keep some free space between the circles
                    boolean add = false;
                    if (dist <= 5) add = true;
                    if (dist >= 8 && dist <= 12) add = true;
                    if (xv <= -1.0 || xv >= 1.0 || yv <= -1.0 || yv >= 1.0) add = false;
                    
                    //Add to data file
                    if (add)
                    {
                        o.write(df.format(xv) + "," + df.format(yv) + "," + cat + "\n");
                    }
                }
            }
            o.close();
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
        System.out.println("done");
    }
    
    public static void fix2()
    {
        
        try
        {
            System.out.println("Fixing set");
            
            BufferedReader in = new BufferedReader(new FileReader("data/g2-2-30.txt"));
            BufferedWriter out = new BufferedWriter(new FileWriter("data/gaussian.csv"));
            out.write("x,y,class\n");
            int n = 1;
            
            String line = in.readLine();
            while (line != null)
            {
                String[] t = line.trim().split(" ");
                if (t.length == 6)
                {
                    System.out.println(t[0] + "," + t[5]);
                }
                int cat = 0;
                if (n > 1024) cat = 1;
                
                out.write(t[0] + "," + t[5] + "," + cat + "\n");
                
                n++;
                line = in.readLine();
            }
            
            in.close();
            out.close();
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
        System.out.println("done");
    }
    
    public static void fix()
    {
        try
        {
            System.out.println("Fixing set");
            DecimalFormat df = new DecimalFormat("0.00"); 
            
            BufferedReader in = new BufferedReader(new FileReader("data/mnist_test.csv"));
            BufferedWriter out = new BufferedWriter(new FileWriter("data/mnist_test2.csv"));
            
            //Header
            out.write("MNIST hand-written characters dataset (28x28 pixels per image)\n");
            
            String line = in.readLine();
            while (line != null)
            {
                String cat = line.substring(0, line.indexOf(","));
                String data = line.substring(line.indexOf(",") + 1, line.length());
                
                String nline = data + "," + cat;
                
                out.write(nline + "\n");
                
                line = in.readLine();
            }
            
            in.close();
            out.close();
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
        System.out.println("done");
    }
}
