import java.util.ArrayList;

public class NearestNeighborHandler extends SupervisedLearner {


    NearestNeigborNode nodes;

    @Override
    public void train(Matrix features, Matrix labels) throws Exception {

        nodes = new NearestNeigborNode();

        ArrayList<double[]> attributes = new ArrayList<>();
        ArrayList<double[]> answers = new ArrayList<>();

        for(int i = 0; i < features.rows(); i++) {
            attributes.add(features.row(i));
            answers.add(labels.row(i));
        }

        nodes.train(attributes, answers, 3, true);

    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {


        labels[0] = nodes.predict(features);


    }


    public double predictionWithSquareError(double[] features, double target) {


        return Math.pow(nodes.predictRegression(features) - target, 2);

    }


    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-N", "-L", "knn", "-A", "credit.arff", "-E", "random", ".75"};
        //args = new String[]{"-L", "neuralnet", "-A", "iris.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }

    }

}
