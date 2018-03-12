import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class DecisionTreeNode {

    Map<Double, DecisionTreeNode> children;
    boolean terminal;
    double terminalValue;
    Map<Double, Double> probabilityProfile;

    Map<Double, Integer> answerCount;
    Map<Double, ArrayList<double[]>> sorting;
    Map<Double, ArrayList<double[]>> sortingAnswers;

    int whichFeature;


    public DecisionTreeNode() {
        alreadyPruned = false;
        alreadyCounted = false;
    }

    public void learn(ArrayList<double[]> matrix, ArrayList<double[]> answers) {

        //find out information for each attribute
        feature bestFeature = null;

        //assign terminal value.  Helps with Pruning.
        answerCount = new HashMap<>();
        for(int i = 0; i < answers.size(); i++) {
            answerCount.putIfAbsent(answers.get(i)[0], 0);
            answerCount.replace(answers.get(i)[0], answerCount.get(answers.get(i)[0]) + 1);
        }

        int bestSoFar = 0;
        for(Map.Entry<Double, Integer> e:answerCount.entrySet()) {
            if(e.getValue() > bestSoFar) {
                bestSoFar = e.getValue();
                terminalValue = e.getKey();
            }
        }
        if(bestSoFar == matrix.size()) {
            terminal = true;
            return;
        }


        for(int i = 0; i < matrix.get(0).length; i++) {


            feature temp = new feature(matrix, answers, i);

            if(bestFeature == null || bestFeature.information > temp.information) {
                bestFeature = temp;
            }

        }
        whichFeature = (int) bestFeature.columnNumber;

        //ok, we have the best feature, now we split it up and make babies, or terminal if there are no more than one option

        if(bestFeature.numberOfFeatures == 1) {
            terminal = true;
            //pick the greatest number of the output class
            terminalValue = bestFeature.biggestOutputClass;
            return;
        }


        sorting = new HashMap<>();
        sortingAnswers = new HashMap<>();
        children = new HashMap<>();

        for(int i = 0; i < matrix.size(); i++) {

            sorting.putIfAbsent(matrix.get(i)[whichFeature], new ArrayList<>());
            sorting.get(matrix.get(i)[whichFeature]).add(matrix.get(i));

            sortingAnswers.putIfAbsent(matrix.get(i)[whichFeature], new ArrayList<>());
            sortingAnswers.get(matrix.get(i)[whichFeature]).add(answers.get(i));

        }
        probabilityProfile = new HashMap<>();

        for(Map.Entry<Double, ArrayList<double[]>> entry:sorting.entrySet()) {
            //at this point, you make the probability profile of answers, but more on that later
            probabilityProfile.putIfAbsent(entry.getKey(), ((double)entry.getValue().size()) / matrix.size());

            children.put(entry.getKey(), new DecisionTreeNode());
            children.get(entry.getKey()).learn(entry.getValue(), sortingAnswers.get(entry.getKey()));
        }


    }



    boolean alreadyCounted;

    public void resetCount() {
        if(terminal) {
            return;
        }
        alreadyCounted = false;
        for(DecisionTreeNode child:children.values()) {
            child.resetCount();
        }
    }

    public int getNodeCount() {
        if(terminal || alreadyCounted) {
            return 0;
        }

        int amount = 0;
        for(DecisionTreeNode child: children.values()) {
            amount += child.getNodeCount();
        }
        alreadyCounted = true;
        return amount + 1;
    }

    public int bestTreeDepth() {
        if(terminal || alreadyCounted) {
            return 0;
        }
        int best = 0;
        for(DecisionTreeNode child: children.values()) {
            int temp = child.bestTreeDepth();
            if(temp > best) {
                best = temp;
            }
        }
        alreadyCounted = true;
        return best + 1;

    }

    boolean alreadyPruned;

    public void combinationPruning(ArrayList<double[]> matrix, ArrayList<double[]> answers) {

        //for each child, compare it against each other child
        if(alreadyPruned || terminal || children.keySet().size() < 3) {
            return;
        }
        alreadyPruned = true;


        Map<Double, DecisionTreeNode> newChildren = new HashMap<>();

        for(double child1:children.keySet()) {

            for(double child2:children.keySet()) {
                if(child1 >= child2) {
                    //we have already computed this comparison, or they are comparing against themselves
                    continue;
                }

                if(children.get(child1).whichFeature == children.get(child2).whichFeature) {
                    //consider combining them
                    DecisionTreeNode combinedChild = new DecisionTreeNode();

                    //separate the data into the pats we would use normally
                    ArrayList<double[]> newData = new ArrayList<>();
                    ArrayList<double[]> newAnswers = new ArrayList<>();

                    for(int i = 0; i < sorting.get(child1).size(); i++) {
                        newData.add(sorting.get(child1).get(i));
                        newAnswers.add(sortingAnswers.get(child1).get(i));
                    }
                    for(int i = 0; i < sorting.get(child2).size(); i++) {
                        newData.add(sorting.get(child2).get(i));
                        newAnswers.add(sortingAnswers.get(child2).get(i));
                    }

                    combinedChild.learn(newData, newAnswers);
                    newChildren.put(child1, combinedChild);
                    newChildren.put(child2, combinedChild);

                }
            }

            if(newChildren.get(child1) == null) {
                newChildren.put(child1, children.get(child1));
            }
        }
        children = newChildren;

        //now, combination prune each child
        for(DecisionTreeNode child:children.values()) {
            child.combinationPruning(matrix, answers);
        }

    }

    public double predictWithNewChildren(double[] features, Map<Double, DecisionTreeNode> newChildren) {

        //will have to do something different for unknown values, but for now this works great

        if(terminal) {
            return terminalValue;
        }

        DecisionTreeNode temp = newChildren.get(features[whichFeature]);
        if(temp == null) {
            //missing value, start the probability profile

            probabilityProfileHolder bestProbability = null;

            for(Map.Entry<Double, DecisionTreeNode> child:newChildren.entrySet()) {

                probabilityProfileHolder p = child.getValue().predictWithProbability(features);
                p.percentage = p.percentage * probabilityProfile.get(child.getKey());
                if(bestProbability == null || p.percentage > bestProbability.percentage) {
                    bestProbability = p;
                }
            }

            return bestProbability.value;
        }
        return temp.predict(features);

    }


    //returns the percent accuracy that you achieve
    public void prune(ArrayList<double[]> matrix, ArrayList<double[]> answers) {

        //base case, can't prune anything at this level
        if(terminal) {
            return;
        }

        //otherwise, we have some recursion to do. send the prune command to the children
        //then we must see if we need to be pruned. We do that by seeing if we can turn ourselves into
        //a terminal node with no loss in accuracy

        for(DecisionTreeNode d:children.values()) {
            d.prune(matrix, answers);
        }


        //no we can be sure our children are pruned, now we check ourselves.
        double amountCorrect = 0;

        for(int i = 0; i < answers.size(); i++) {
            if(answers.get(i)[0] == terminalValue) {
                amountCorrect++;
            }
        }

        double selfAccuracy = amountCorrect / answers.size();

        double correctCount = 0;
        for(int i = 0; i < matrix.size(); i++) {
            if(predict(matrix.get(i)) == answers.get(i)[0]) {
                correctCount++;
            }
        }
        if(selfAccuracy >= correctCount / matrix.size()) {
            //we are no better than terminal, prune ourselves
            children = null;
            terminal = true;
        }
    }






    public double predict(double[] features) {

        //will have to do something different for unknown values, but for now this works great

        if(terminal) {
            return terminalValue;
        }

        DecisionTreeNode temp = children.get(features[whichFeature]);
        if(temp == null) {
            //missing value, start the probability profile

            probabilityProfileHolder bestProbability = null;

            for(Map.Entry<Double, DecisionTreeNode> child:children.entrySet()) {

                probabilityProfileHolder p = child.getValue().predictWithProbability(features);
                p.percentage = p.percentage * probabilityProfile.get(child.getKey());
                if(bestProbability == null || p.percentage > bestProbability.percentage) {
                    bestProbability = p;
                }
            }

            return bestProbability.value;
        }
        return temp.predict(features);
    }


    public probabilityProfileHolder predictWithProbability(double[] features) {

        if(terminal) {
            return new probabilityProfileHolder(1, terminalValue);
        }

        DecisionTreeNode temp = children.get(features[whichFeature]);
        if(temp == null) {
            //missing value, start the probability profile

            probabilityProfileHolder bestProbability = null;

            for(Map.Entry<Double, DecisionTreeNode> child:children.entrySet()) {

                probabilityProfileHolder p = child.getValue().predictWithProbability(features);
                p.percentage = p.percentage * probabilityProfile.get(child.getKey());
                if(bestProbability == null || p.percentage > bestProbability.percentage) {
                    bestProbability = p;
                }
            }

            return bestProbability;
        }
        return temp.predictWithProbability(features);

    }



    class probabilityProfileHolder {
        double percentage;
        double value;

        public probabilityProfileHolder(double percentage, double value) {
            this.percentage = percentage;
            this.value = value;
        }
    }






    class feature {

        Map<Double, Integer> featureCount;
        Map<Double, Map<Double, Integer>> outputCount;

        double information;
        int numberOfFeatures;
        double columnNumber;
        double biggestOutputClass;
        int biggestNumberOfOutputClass;

        public feature(ArrayList<double[]> matrix, ArrayList<double[]> answers, int column) {

            columnNumber = column;
            information = 0;
            numberOfFeatures = 0;
            featureCount = new HashMap<>();
            outputCount = new HashMap<>();
            biggestNumberOfOutputClass = 0;


            //create the information here

            for(int i = 0; i < matrix.size(); i++) {

                double temp = matrix.get(i)[column];

                featureCount.putIfAbsent(temp, 0);
                featureCount.replace(temp, featureCount.get(temp) + 1);

                //now update the output count

                outputCount.putIfAbsent(temp, new HashMap<>());
                outputCount.get(temp).putIfAbsent(answers.get(i)[0], 0);
                outputCount.get(temp).replace(answers.get(i)[0], outputCount.get(temp).get(answers.get(i)[0]) + 1);

            }

            //ok, we have initialized all of the counts, now we go through it

            for(Map.Entry<Double, Integer> feature: featureCount.entrySet()) {
                numberOfFeatures++;

                double insideValue = 0;

                for(Map.Entry<Double, Integer> output: outputCount.get(feature.getKey()).entrySet()) {

                    if(output.getValue() > biggestNumberOfOutputClass) {
                        biggestOutputClass = output.getKey();
                    }

                    double temp = ((double) output.getValue()) / ((double) feature.getValue());

                    insideValue -= Math.log(temp)/Math.log(2) * temp;

                }
                information += ((double) feature.getValue()) / ( (double) matrix.size()) * insideValue;

            }

        }


    }


}
