public class Perceptron_v1 {

    private final double learningRate = .1;

    private final double theta = 1;
    private double thetaWeight;

    private int numberOfInputs;
    private double[] weights;


    public Perceptron_v1(int numberOfInputs) {
        weights = new double[numberOfInputs];
        this.numberOfInputs = numberOfInputs;
    }

    public void learn(double[] inputs, boolean expected) {
        if(inputs.length != numberOfInputs) {
            System.out.println("Error, number of inputs does not match number of weights");
            return;
        }

        if(test(inputs) != expected) {
            //modify weights, as the test failed

            //first, find out if we need to go up or down

            int direction = -1;

            if(expected) {
                direction = 1;
            }

            for(int i = 0; i < numberOfInputs; i++) {
                weights[i] += learningRate * direction * inputs[i];
            }
            thetaWeight += learningRate * direction * theta;


        }

    }

    public boolean test(double[] inputs) {

        double total = 0;
        for(int i = 0; i < numberOfInputs; i++) {
            total += inputs[i] * weights[i];
        }
        total += theta * thetaWeight;

        return (total > 0);
    }


    public double testWithConfidence(double[] inputs) {
        double total = 0;
        for(int i = 0; i < numberOfInputs; i++) {
            total += inputs[i] * weights[i];
        }
        total += theta * thetaWeight;

        return total;
    }



}
