import java.util.*;

public class NearestNeigborNode {

    ArrayList<Node> data;
    int k;
    boolean weighted;
    final boolean[] continuousAttributes = {false, true, true, false, false, false, false, true, false, false, true, false, false, true, true};

    List<Map<Double, Double>> masterLookup;
    double totalValues;

    public void train(ArrayList<double[]> features, ArrayList<double[]> labels, int k, boolean w) {


        data = new ArrayList<>();
        this.k = k;
        weighted = w;
        masterLookup = new LinkedList<>();
        totalValues = features.size();

        for(int i = 0; i < features.size(); i++) {
            data.add(new Node(features.get(i), labels.get(i)));
        }

        for(int i = 0; i < continuousAttributes.length; i++) {
            //cycle through attribute columns
            masterLookup.add(new HashMap<>());
            if(continuousAttributes[i]) {
                continue;
            }
            Map<Double, Double> masterMap = masterLookup.get(i);

            for(int j = 0; j < features.size(); j++) {

                masterMap.putIfAbsent(features.get(j)[i], 0.0);
                if(labels.get(j)[0] == 0) {
                    masterMap.replace(features.get(j)[i], masterMap.get(features.get(j)[i]) - 1);
                } else {
                    masterMap.replace(features.get(j)[i], masterMap.get(features.get(j)[i]) + 1);
                }
            }


        }




    }


    public double predict(double[] features) {


        topKNodes answer = new topKNodes(k, weighted);
        for(Node n:data) {
            n.setDistance(features);
            answer.add(n);
        }

        return answer.classify();
    }

    public double predictRegression(double[] features) {
        topKNodes answer = new topKNodes(k, weighted);
        for(Node n:data) {
            n.setDistance(features);
            answer.add(n);
        }

        return answer.classifyRegression();
    }



    class topKNodes {

        int k;
        ArrayList<Node> nodes;
        double worstDistance;
        boolean weighting;

        public topKNodes(int k, boolean w) {
            this.k = k;
            nodes = new ArrayList<>();
            weighting = w;
        }

        public void add(Node n) {

            if(n.distance < worstDistance && nodes.size() == k) {
                //pop off the worst node
                nodes.remove(0);
            } else if(n.distance >= worstDistance && nodes.size() == k) {
                return;
            }
            //add in the node where it should go
            if(nodes.size() == 0) {
                nodes.add(n);
                worstDistance = n.distance;
                return;
            }

            for(int i = 0; i < nodes.size(); i++) {
                if(n.distance > nodes.get(i).distance) {
                    nodes.add(i, n);
                    worstDistance = nodes.get(0).distance;
                    return;
                }
            }
            nodes.add(n);
            worstDistance = nodes.get(0).distance;
        }





        public double classify() {
            Map<Double, Double> classifyConfidence = new HashMap<>();

            for(int i = 0; i < nodes.size(); i++) {

                Node n = nodes.get(i);

                classifyConfidence.putIfAbsent(n.classification, 0.0);
                if(weighting) {
                    classifyConfidence.replace(n.classification, classifyConfidence.get(n.classification) + 1.0 / Math.pow(n.distance, 2));
                } else {
                    classifyConfidence.replace(n.classification, classifyConfidence.get(n.classification) + 1);
                }
            }

            double bestClassification = 0;
            double bestConfidence = 0;
            for(Map.Entry<Double, Double> entry: classifyConfidence.entrySet()) {
                if(entry.getValue() > bestConfidence) {
                    bestConfidence = entry.getValue();
                    bestClassification = entry.getKey();
                }
            }


            return bestClassification;
        }





        public double classifyRegression() {

            double currentValue = 0;
            double totalDistance = 0;

            for(Node n:nodes) {

                if(weighting) {
                    currentValue += n.classification / Math.pow(n.distance, 2);
                    totalDistance += 1.0 / Math.pow(n.distance, 2);
                } else {
                    currentValue += n.classification;
                    totalDistance += n.distance;
                }
            }
            if(weighting) {
                return currentValue / totalDistance;
            } else {
                return currentValue / k;
            }
        }
    }


    class Node {

        double[] features;
        double classification;
        double distance;

        public Node(double[] features, double[] labels) {
            this.features = features;
            classification = labels[0];
        }

        public void setDistance(double[] inputs) {
            distance = 0;

            for(int i = 0; i < features.length; i++) {

                if(inputs[i] >= 1000) {
                    distance += 1;
                } else if(continuousAttributes[i]) {



                    distance += Math.abs(features[i] - inputs[i]);
                } else {
                    double value1 = masterLookup.get(i).get(features[i]);
                    double value2 = masterLookup.get(i).get(inputs[i]);
                    double ratio = Math.abs(value1 - value2) / totalValues;
                    distance += ratio;
                }
            }
        }
    }

}
