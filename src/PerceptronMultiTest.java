import java.io.File;
import java.util.Scanner;

public class PerceptronMultiTest {

    public static void main(String args[]) {

        String fileName = "SampleData.txt";

        File file = new File(fileName);

        sampleDataCreation.createData(file);

        Scanner sc = null;

        final int numberOfInputs = 2;




        Perceptron_v1 perceptron = new Perceptron_v1(2);


        while(true) {

            try {
                sc = new Scanner(file);
            } catch (Exception e) {

            }


            double[] deltaTheta = perceptron.weights.clone();
            double thetaWeight = perceptron.thetaWeight;


            double[] inputs = new double[numberOfInputs];

            try {

                while (true) {

                    for (int k = 0; k < numberOfInputs; k++) {

                        inputs[k] = sc.nextDouble();
                    }

                    perceptron.learn(inputs, sc.nextBoolean());

                }
            } catch(Exception e) {

            }

            double total = 0;

            for(int i = 0; i < deltaTheta.length; i++) {
                total += Math.abs(deltaTheta[i] - perceptron.weights[i]);
            }
            total += Math.abs(thetaWeight - perceptron.thetaWeight);


            if(total < .5) {
                break;
            }



        }


        //for now, lets do manual testing


        Scanner reader = new Scanner(System.in);

        while(true) {

            double inputs[] = new double[numberOfInputs];

            for(int k = 0; k < numberOfInputs; k++) {

                inputs[k] = reader.nextDouble();

            }
            System.out.println(perceptron.test(inputs));


        }



    }



}
