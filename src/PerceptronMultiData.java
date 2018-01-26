import java.util.Random;

public class PerceptronMultiData extends SupervisedLearner {




    Random rand;
    Perceptron_v1 perp1;
    Perceptron_v1 perp2;
    Perceptron_v1 perp3;


    public PerceptronMultiData(Random rand) {
        this.rand = rand;

    }


    @Override
    public void train(Matrix features, Matrix labels) throws Exception {


        int seed;
        int numberOfInputs = features.cols();
        perp1 = new Perceptron_v1(numberOfInputs);
        perp2 = new Perceptron_v1(numberOfInputs);
        perp3 = new Perceptron_v1(numberOfInputs);


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
                double expected = labels.get(i, 0);
                boolean result1 = false;
                boolean result2 = false;
                boolean result3 = false;
                if (expected == 0) {
                    result1 = true;
                } else if(expected == 1) {
                    result2 = true;
                } else {
                    result3 = true;
                }
                perp1.learn(features.row(i), result1);
                perp2.learn(features.row(i), result2);
                perp3.learn(features.row(i), result3);

            }


            //testing accuracy
            for (int i = 0; i < features.rows(); i++) {
                int result;
                double confidence1 = perp1.testWithConfidence(features.row(i));
                double confidence2 = perp1.testWithConfidence(features.row(i));
                double confidence3 = perp1.testWithConfidence(features.row(i));

                if(confidence1 > confidence2) {
                    if(confidence1 > confidence3) {
                        result = 0;
                    } else {
                        result = 2;
                    }
                } else {
                    if(confidence2 > confidence3) {
                        result = 1;
                    } else {
                        result = 2;
                    }
                }

                accuracy += (result == labels.get(i, 0)) ? 1 : 0;
            }
            accuracy = accuracy / features.rows();

            //System.out.println("Epoch: " + totalRepititions + "  Accuracy: " + accuracy);


        }

        System.out.println("total repititions: " + totalRepititions);

    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        double confidence1 = perp1.testWithConfidence(features);
        double confidence2 = perp2.testWithConfidence(features);
        double confidence3 = perp3.testWithConfidence(features);

        if(confidence1 > confidence2) {
            if(confidence1 > confidence3) {
                labels[0] = 0;
            } else {
                labels[0] = 2;
            }
        } else {
            if(confidence2 > confidence3) {
                labels[0] = 1;
            } else {
                labels[0] = 2;
            }
        }
        //System.out.println(labels[0]);

    }


    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-L", "perceptronMulti", "-A", "iris.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }

    }
}
