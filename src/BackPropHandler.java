import java.util.ArrayList;
import java.util.Random;

public class BackPropHandler extends SupervisedLearner {

    private Random rand;

    private final int[] numberOfNodes = {8, 3};    //used to initialize the amount of nodes per layer.  last index is output layer
    private int neuralNetLength;
    private BackPropLayer network;




    public BackPropHandler(Random rand) {
        this.rand = rand;
        neuralNetLength = numberOfNodes.length;
    }



    @Override
    public void train(Matrix features, Matrix labels) throws Exception {

        //first things first, create the layer network
        int numberOfInputs = features.cols();

        network = new BackPropLayer(numberOfInputs, numberOfNodes[0], rand);
        BackPropLayer temp;
        BackPropLayer current = network;
        for(int i = 1; i < numberOfNodes.length; i++) {
            temp = new BackPropLayer(numberOfNodes[i - 1], numberOfNodes[i], rand);
            current.addBackPropLayer(temp);
            current = temp;
        }

        features.shuffle(rand, labels);

        int reps = 0;
        int repsSinceBest = 0;
        double previousAccuracy = 0;
        int testSet = features.rows() / 4;

        while(reps < 1000 && repsSinceBest < 8) {

            reps++;
            repsSinceBest++;

            //calculate Accuracy, 25% rule
            double accuracy = 0;
            for(int i = 0; i < testSet; i++) {
                if(labels.get(i, 0) == sanePrediction(features.row(i))) {
                    accuracy++;
                }
            }
            accuracy = accuracy / testSet;
            if(accuracy > previousAccuracy) {
                previousAccuracy = accuracy;
                repsSinceBest = 0;
            }



            // loop through the inputs
            for (int i = testSet; i < features.rows(); i++) {

                //format the output, works great for multiple output nodes
                double[] formattedAnswers = new double[numberOfNodes[numberOfNodes.length - 1]];
                for(int x = 0; x < formattedAnswers.length; x++) {
                    if(labels.row(i)[0] == x) {
                        formattedAnswers[x] = 1;
                    } else {
                        formattedAnswers[x] = 0;
                    }
                }

                //pass it in to the first layer, have it work itself out
                network.learn(features.row(i), formattedAnswers);

            }
        }

        //System.out.println(network.toString());
        System.out.println("Number of Reps: " + reps);

    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        double[] temp = network.predict(features);

        int bestAnswer = 0;
        double bestConfidence = 0;

        for(int i = 0; i < temp.length; i++) {
            if(bestConfidence < temp[i]) {
                bestAnswer = i;
                bestConfidence = temp[i];
            }
        }
        labels[0] = bestAnswer;
    }

    public int sanePrediction(double[] features) {
        double[] temp = network.predict(features);

        int bestAnswer = 0;
        double bestConfidence = 0;

        for(int i = 0; i < temp.length; i++) {
            if(bestConfidence < temp[i]) {
                bestAnswer = i;
                bestConfidence = temp[i];
            }
        }
        return bestAnswer;
    }




    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-L", "neuralnet", "-A", "iris.arff", "-E", "random", ".75"};
        //args = new String[]{"-L", "neuralnet", "-A", "iris.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }

    }









}