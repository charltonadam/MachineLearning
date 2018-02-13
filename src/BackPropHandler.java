import java.util.ArrayList;
import java.util.Random;

public class BackPropHandler extends SupervisedLearner {

    Random rand;

    final int[] numberOfNodes = {1};    //used to initialize the amount of nodes per layer.  last index is output layer
    ArrayList<BackPropNode>[] neuralNet;
    int neuralNetLength;




    public BackPropHandler(Random rand) {
        this.rand = rand;
        neuralNet = new ArrayList[numberOfNodes.length];
        neuralNetLength = numberOfNodes.length;


    }



    @Override
    public void train(Matrix features, Matrix labels) throws Exception {

        int numberOfInputs = features.cols();



        //initialize the neural net
        for(int i = 0; i < neuralNetLength; i++) {
            neuralNet[i] = new ArrayList<>();

            for(int j = 0; j < numberOfNodes[i]; i++) {
                neuralNet[i].add(new BackPropNode(i == 0 ? numberOfInputs: neuralNet[i - 1].size(), rand));
            }
        }


        //now train on the feature set

        //first loop, repetition of lines of data
        for(int i = 0; i < features.rows(); i++) {

            double[] input = features.row(i);


            //next repetition, each layer in the neural net
            for(int k = 0; k < neuralNetLength; k++) {

                double[] output;

                if(k == neuralNetLength - 1) {
                    if(k != 0) {
                        output = new double[ neuralNet[k - 1].size()];
                    } else {
                        //there is no hidden layers, so it doesn't really matter if we keep the input or not
                        output = new double[numberOfInputs];
                    }

                } else {
                    output = new double[neuralNet[k + 1].size()];
                }

                //next repetition, each node in the layer
                for(int x = 0; x < neuralNet[k].size(); x++) {
                    if(k == neuralNetLength - 1) {
                        //we are on the output layer, use the different technique

                        double[] error = neuralNet[k].get(x).predictWithError(input, labels.get(i,x));
                        for(int z = 0; z < error.length; z++) {
                            output[z] += error[z];
                        }
                    }


                    output[x] =  neuralNet[k].get(x).predict(input);

                }

                //output has been initialized, move it to the next layer
                input = output;

            }

            //at this point, the error is contained within the "input" array





            //TODO: this whole section needs to be rethought
            //prop the errors back through the layers
            for(int k = neuralNetLength - 2; k >= 0; k--) {

                double[] error;

                if(k == 0) {
                    error = new double[neuralNet[k].size()];
                } else {
                    error = new double[neuralNet[k-1].size()];
                }


                //prop the errors for each node in the layer
                for(int x = neuralNet[k].size() - 1; x >= 0; x--) {

                    double[] adjustment = neuralNet[k].get(x).adjustForError(input[x]);

                }

            }


        }






    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

    }




    public static void main(String args[]) {



    }
}