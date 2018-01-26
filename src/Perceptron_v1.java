public class Perceptron_v1 {

    final double learningRate = .1;

    final double theta = 1;
    public double thetaWeight;

    int numberOfInputs;
    public double[] weights;


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

        if(total > 0) {
            return true;
        }
        return false;
    }



}
