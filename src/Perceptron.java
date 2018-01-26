import java.util.Random;

public class Perceptron extends SupervisedLearner {

    Random rand;
    Perceptron_v1 perp;


    public Perceptron(Random rand) {
        this.rand = rand;

    }


    @Override
    public void train(Matrix features, Matrix labels) throws Exception {


        if(true) {   //this is a manual switch for running data with the IRIS data set
            //inside here is standard data, only binary output


            int seed;
            int numberOfInputs = features.cols();
            perp = new Perceptron_v1(numberOfInputs);

            int totalRepititions = 0;
            int repititionsSinceChange = 0;
            double accuracy = 0;
            double previousAccuracy = 0;


            while (repititionsSinceChange < 5 && totalRepititions < 1000) {


                seed = rand.nextInt();

                features.shuffle(new Random(seed));
                labels.shuffle(new Random(seed));


                if (previousAccuracy - accuracy < -.05) {
                    repititionsSinceChange = 0;
                } else {
                    repititionsSinceChange++;
                }

                previousAccuracy = accuracy;
                accuracy = 0;
                totalRepititions++;


                for (int i = 0; i < features.rows(); i++) {
                    boolean result = false;
                    if (labels.get(i, 0) == 1) {
                        result = true;
                    }
                    perp.learn(features.row(i), result);
                }

                for (int i = 0; i < features.rows(); i++) {
                    boolean test = false;

                    if (labels.get(i, 0) == 1) {
                        test = true;
                    }

                    accuracy += (perp.test(features.row(i)) == test) ? 1 : 0;
                }
                accuracy = accuracy / features.rows();

                //System.out.println("Epoch: " + totalRepititions + "  Accuracy: " + accuracy);


            }

            System.out.println("total repititions: " + totalRepititions);



        } else {
            //This is for the IRIS dataset, triple output, which means we need to have 3 Perceptrons

            



        }

    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        labels[0] =  perp.test(features) ? 1 : 0;

    }


    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-L", "perceptron", "-A", "testNonLineSep.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }

    }





}
