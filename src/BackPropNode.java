import java.util.Random;

public class BackPropNode {

    double[] weights;
    int numberOfInputs;
    final double learningRate = .1;

    double previousOutput;
    double[] previousInputs;
    double[] previousDeltaWeights;
    final double momentumAmount = 0;

    public BackPropNode(int numberOfInputs, Random rand) {

        weights = new double[numberOfInputs + 1]; //bias weight is included in the array, much easier
        this.numberOfInputs = numberOfInputs;

        previousDeltaWeights = new double[numberOfInputs + 1];

        //initialize gaussian weights using random.  Might be slightly too large, keep track of potential problems (takes too long to learn)
        for(int i = 0; i < numberOfInputs + 1; i++) {
            weights[i] = rand.nextGaussian() * .5;
        }
    }

    public double predict(double[] inputs) {

        previousInputs = inputs;

        if(inputs.length != numberOfInputs) {
            System.out.println("Error, number of inputs does not match number of weights");
            return 0;
        }

        double total = 0;

        for(int i = 0; i < numberOfInputs; i++) {
            total += inputs[i] * weights[i];
        }
        total += weights[numberOfInputs];
        previousOutput = 1.0/(1.0 + Math.pow(Math.E, total * -1));

        return previousOutput;

    }


    public double[] predictWithError(double[] inputs, double target) {

        double[] errorVector = new double[numberOfInputs];

        double output = predict(inputs);
        double error = (target - output) * output * (1 - output);

        for(int i = 0; i < numberOfInputs; i++) {
            errorVector[i] = error * weights[i];
            weights[i] += learningRate * error * inputs[i];
        }
        weights[numberOfInputs] += learningRate * error;

        return errorVector;
    }

    public double[] adjustForError(double netError) {

        double[] errorVector = new double[numberOfInputs];
        double error = previousOutput * (1 - previousOutput) * netError;

        for(int i = 0; i < numberOfInputs; i++) {
            errorVector[i] = error * weights[i];
            double deltaWeight = learningRate * error * previousInputs[i];
            weights[i] += deltaWeight + momentumAmount * previousDeltaWeights[i];
            previousDeltaWeights[i] = deltaWeight;
        }
        double deltaWeight = learningRate * error;
        weights[numberOfInputs] += deltaWeight + momentumAmount * previousDeltaWeights[numberOfInputs];
        previousDeltaWeights[numberOfInputs] = deltaWeight;

        return errorVector;


    }

    public String toString() {

        String ret = "";

        for(int i = 0; i < numberOfInputs + 1; i++) {
            ret += weights[i] + "\n";
        }
        return ret;

    }

}
