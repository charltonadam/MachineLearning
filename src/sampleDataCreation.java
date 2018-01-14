import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

public class sampleDataCreation {


    public static void createData(File file) {


        Random rand = new Random();
        PrintWriter writer = null;

        try {
            writer = new PrintWriter(file);
        } catch(Exception e) {

        }

        for(int i = 0; i < 10000; i++) {


            int x = rand.nextInt(50);
            int y = rand.nextInt(50);
            boolean check = x > y;

            writer.print(x + " " + y + " " + check + "\n");
        }




    }




}
