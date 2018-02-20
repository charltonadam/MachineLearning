import java.util.Random;

public class BackPropLayer {

    boolean isOutputLayer = true;
    BackPropLayer next;
    int numberOfNodes;
    int numberOfInputs;

    BackPropNode[] nodes;



    public BackPropLayer(int numberOfInputs, int numberOfNodes, Random rand) {
        this.numberOfInputs = numberOfInputs;
        this.numberOfNodes = numberOfNodes;

        nodes = new BackPropNode[numberOfNodes];

        for(int i = 0; i < numberOfNodes; i++) {
            nodes[i] = new BackPropNode(numberOfInputs, rand);
        }

    }

    public void addBackPropLayer(BackPropLayer next) {
        this.next = next;
        isOutputLayer = false;
    }


    //returns the error  vector
    public double[] learn(double[] inputs, double[] answers) {

        if(isOutputLayer) {

            double[] returnValue = new double[numberOfInputs];

            for(int i = 0; i < numberOfNodes; i++) {
                double[] error = nodes[i].predictWithError(inputs, answers[i]);

                for(int x = 0; x < numberOfInputs; x++) {
                    returnValue[x] += error[x];
                }
            }

            return returnValue;

        } else {


            double[] output = new double[numberOfNodes];


            for(int i = 0; i < numberOfNodes; i++) {
                output[i] = nodes[i].predict(inputs);
            }


            double[] error = next.learn(output, answers);
            double[] retValue = new double[numberOfInputs];


            for(int i = 0; i < numberOfNodes; i++) {
                double[] temp = nodes[i].adjustForError(error[i]);

                for(int x = 0; x < numberOfInputs; x++) {
                    retValue[x] += temp[x];
                }
            }

            return retValue;
        }
    }


    public double[] predict(double[] inputs) {

        double[] output = new double[numberOfNodes];

        for(int i = 0; i < numberOfNodes; i++) {
            output[i] = nodes[i].predict(inputs);
        }

        if(isOutputLayer) {
            return output;
        } else {
            return next.predict(output);
        }

    }

    public String toString() {
        String ret = "";
        ret += "nextLayer\n";

        for(int i = 0; i < numberOfNodes; i++) {
            ret += "Node: " + i + "\n";
            ret += nodes[i].toString();
            ret += "\n";
        }
        if(isOutputLayer) {
            return ret;
        }
        ret += next.toString();
        return ret;
    }



}
