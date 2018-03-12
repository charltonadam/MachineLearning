import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

public class BackPropHandler extends SupervisedLearner {

    private Random rand;

    private final int[] numberOfNodes = {20, 4};    //used to initialize the amount of nodes per layer.  last index is output layer
    private int neuralNetLength;
    private BackPropLayer network;


    public BackPropHandler(Random rand) {
        this.rand = rand;
        neuralNetLength = numberOfNodes.length;
    }



    @Override
    public void train(Matrix features, Matrix labels) throws Exception {


        File outputFile = new File("output.csv");
        PrintWriter writer = null;

        try {
            writer = new PrintWriter(outputFile);
        }catch(Exception ignored) {

        }
        writer.println("Epoch,VS MSE,VS percentage,training MSE");


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
        double previousVSAccuracy = 0;
        int testSet = features.rows() / 4;  //25% for validation set

        while(reps < 10000 && repsSinceBest < 5) {

            reps++;
            repsSinceBest++;

            //calculate Accuracy using Validation Set
            double VSaccuracy = 0;
            double accuracyPercentage = 0;
            for(int i = 0; i < testSet; i++) {
                if(labels.get(i, 0) == sanePrediction(features.row(i))) {
                    accuracyPercentage++;
                }
                VSaccuracy += predictionWithSquareError(features.row(i), labels.get(i, 0));
            }
            accuracyPercentage = accuracyPercentage / testSet;
            VSaccuracy = VSaccuracy / testSet;
            if(VSaccuracy < previousVSAccuracy * .999 || reps < 50) {
                previousVSAccuracy = VSaccuracy;
                repsSinceBest = 0;
            }

            double trainingAccuracy = 0;
            for(int i = testSet; i < features.rows(); i++) {
                trainingAccuracy += predictionWithSquareError(features.row(i), labels.get(i, 0));
            }
            trainingAccuracy = trainingAccuracy / (features.rows() - testSet);

            writer.println(reps + "," + VSaccuracy + "," + accuracyPercentage + "," + trainingAccuracy);

            //System.out.println("Epoch: " + reps);
            //System.out.println("VS Accuracy MSE: " + VSaccuracy);
            //System.out.println("VS Accuracy percentage: ");
            //System.out.println("Training Accuracy MSE: " + trainingAccuracy + "\n");




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
        writer.close();

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

    public double predictionWithSquareError(double[] features, double answer) {

        double[] temp = network.predict(features);

        return Math.pow(temp[(int) answer] - 1, 2);

    }




    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-L", "neuralnet", "-A", "NameData.arff", "-E", "training"};
        //args = new String[]{"-L", "neuralnet", "-A", "iris.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }

    }

    public double measureAccuracy(Matrix features, Matrix labels, Matrix confusion) throws Exception {
        if(features.rows() != labels.rows())
            throw(new Exception("Expected the features and labels to have the same number of rows"));
        if(labels.cols() != 1)
            throw(new Exception("Sorry, this method currently only supports one-dimensional labels"));
        if(features.rows() == 0)
            throw(new Exception("Expected at least one row"));

        int labelValues = labels.valueCount(0);
        if(labelValues == 0) // If the label is continuous...
        {
            // The label is continuous, so measure root mean squared error
            double[] pred = new double[1];
            double sse = 0.0;
            for(int i = 0; i < features.rows(); i++)
            {
                double[] feat = features.row(i);
                double[] targ = labels.row(i);
                pred[0] = 0.0; // make sure the prediction is not biassed by a previous prediction
                predict(feat, pred);
                double delta = targ[0] - pred[0];
                sse += (delta * delta);
            }
            return Math.sqrt(sse / features.rows());
        }
        else
        {
            // The label is nominal, so measure predictive accuracy
            if(confusion != null)
            {
                confusion.setSize(labelValues, labelValues);
                for(int i = 0; i < labelValues; i++)
                    confusion.setAttrName(i, labels.attrValue(0, i));
            }
            int correctCount = 0;
            double[] prediction = new double[1];
            double squareError = 0;
            for(int i = 0; i < features.rows(); i++)
            {
                double[] feat = features.row(i);
                int targ = (int)labels.get(i, 0);
                if(targ >= labelValues)
                    throw new Exception("The label is out of range");
                predict(feat, prediction);
                squareError += predictionWithSquareError(feat, targ);
                int pred = (int)prediction[0];
                if(confusion != null)
                    confusion.set(targ, pred, confusion.get(targ, pred) + 1);
                if(pred == targ)
                    correctCount++;
            }
            squareError = squareError / features.rows();
            //System.out.println("Square Error for Test: " + squareError);
            return (double)correctCount / features.rows();
        }
    }


}