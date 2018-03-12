import java.util.*;

public class DecisionTreeHandler extends SupervisedLearner {

    DecisionTreeNode root;


    @Override
    public void train(Matrix features, Matrix labels) throws Exception {

        Random rand = new Random();
        features.shuffle(rand, labels);

        ArrayList<double[]> matrix = new ArrayList<>();
        ArrayList<double[]> answers = new ArrayList<>();

        ArrayList<double[]> matrixValidation = new ArrayList<>();
        ArrayList<double[]> answersValidation = new ArrayList<>();


        //take 25% off the top for validation set
        /*for(int i = 0; i < features.rows() / 4; i++) {
            matrixValidation.add(features.row(i));
            answersValidation.add(labels.row(i));
        }*/


        for(int i = 0/*features.rows() / 4*/; i < features.rows(); i++) {
            matrix.add(features.row(i));
            answers.add(labels.row(i));
        }

        featureAdder(matrix, answers);

        root = new DecisionTreeNode();
        root.learn(matrix, answers);
        root.combinationPruning(matrix, answers);
        //root.prune(matrixValidation, answersValidation);

        System.out.println("Node Count: " + root.getNodeCount() );
        root.resetCount();
        System.out.println("Tree Depth: " + root.bestTreeDepth());


    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        labels[0] = root.predict(features);

    }



    ArrayList<double[]> featureAdder(ArrayList<double[]> matrix, ArrayList<double[]> answers) {

        Map<Double, Map<Double, Integer>> countOfAttribute;
        Map<Double, Double> bestAttribute;
        Set<Integer> missingIndexes;

        //this is to replace values for each feature, without changing the order
        for(int i = 0; i < matrix.get(0).length; i++) {

            countOfAttribute = new HashMap<>();
            bestAttribute = new HashMap<>();
            missingIndexes = new HashSet<>();

            //for each data point, check to see if the feature is missing, and if so add it to the missingIndexes.
            //otherwise, add the count to countOfAttributes
            for(int k = 0; k < matrix.size(); k++) {

                if(matrix.get(k)[i] > matrix.size()) {
                    //found an unknown
                    missingIndexes.add(k);
                } else {
                    //add it to the attribute count
                    countOfAttribute.putIfAbsent(answers.get(k)[0], new HashMap<>());
                    countOfAttribute.get(answers.get(k)[0]).putIfAbsent(matrix.get(k)[i], 0);
                    int amount = countOfAttribute.get(answers.get(k)[0]).get(matrix.get(k)[i]);
                    countOfAttribute.get(answers.get(k)[0]).replace(matrix.get(k)[i], amount + 1);
                }
            }
            //the counts are initialized, now create the bestAttributes list

            for(Map.Entry<Double, Map<Double, Integer>> e:countOfAttribute.entrySet()) {

                bestAttribute.putIfAbsent(e.getKey(), 0.0);

                int bestCount = 0;

                for(Map.Entry<Double, Integer> s:e.getValue().entrySet()) {
                    if(s.getValue() > bestCount) {
                        bestAttribute.replace(e.getKey(), s.getKey());
                        bestCount = s.getValue();
                    }
                }

            }

            //now, replace all the data in the column with the appropriate data

            for(int k:missingIndexes) {
                matrix.get(k)[i] = bestAttribute.get(answers.get(k)[0]);
            }
        }
        return matrix;

    }


    public static void main(String args[]) {

        MLSystemManager runner = new MLSystemManager();

        args = new String[]{"-L", "decisiontree", "-A", "cars.arff", "-E", "cross", "10"};
        //args = new String[]{"-L", "decisiontree", "-A", "cars.arff", "-E", "training"};
        //args = new String[]{"-L", "neuralnet", "-A", "iris.arff", "-E", "training"};
        try {
            runner.run(args);
        }catch(Exception e) {
            e.printStackTrace();

        }
    }
}
