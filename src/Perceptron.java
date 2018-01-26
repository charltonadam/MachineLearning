import java.util.Random;

public class Perceptron extends SupervisedLearner {

    Random rand;
    Perceptron_v1 perp;


    public Perceptron(Random rand) {
        this.rand = rand;

    }


    @Override
    public void train(Matrix features, Matrix labels) throws Exception {

        int numberOfInputs = features.cols();
        perp = new Perceptron_v1(numberOfInputs);

        double[] savedWeights = perp.weights;
        double savedTheta = perp.thetaWeight;

        int totalRepititions = 0;
        int repititionsSinceChange = 0;

        while(repititionsSinceChange < 5 && totalRepititions < 1000) {

            if(weightCheck(savedWeights, perp.weights, savedTheta, perp.thetaWeight) != 0) {
                repititionsSinceChange = 0;
            } else {
                repititionsSinceChange++;
            }
            totalRepititions++;
            savedWeights = perp.weights;
            savedTheta = perp.thetaWeight;


            for(int i = 0; i < features.rows(); i++) {
                boolean result = false;
                if(labels.get(i, 0) == 1) {
                    result = true;
                }
                perp.learn(features.row(i), result);
            }

        }

        System.out.println("total repititions: " + totalRepititions);

    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        labels[0] =  perp.test(features) ? 1 : 0;

    }



    public double weightCheck(double[] weights1, double[] weights2, double theta1, double theta2) {

        double difference = 0;

        for(int i = 0; i < weights1.length; i++) {
            difference = Math.abs(weights1[i] - weights2[i]);
        }
        difference = Math.abs(theta1 - theta2);


        return difference;
    }





    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-L", "perceptron", "-A", "testLineSep.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }

    }





}
