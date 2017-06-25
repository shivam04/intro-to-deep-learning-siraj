from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights1 = 2 * random.random((3, 5)) - 1
        self.synaptic_weights2 = 2 * random.random((5, 4)) - 1
        self.synaptic_weights3 = 2 * random.random((4, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            hidden_l1 = self.think(training_set_inputs,self.synaptic_weights1)
            hidden_l2 = self.think(hidden_l1,self.synaptic_weights2)
            output = self.think(hidden_l2,self.synaptic_weights3)
          #  print "h1",hidden_l1.shape
           # print "h2",hidden_l2.shape
           # print "o",output.shape
            del4 = (training_set_outputs - output)*self.__sigmoid_derivative(output)
           # print "del4",del4.shape
            del3 = dot(self.synaptic_weights3,del4.T)*(self.__sigmoid_derivative(hidden_l2).T)
            del2 = dot(self.synaptic_weights2,del3)*(self.__sigmoid_derivative(hidden_l1).T)
            adjustment3 = dot(hidden_l2.T, del4)
            adjustment2 = dot(hidden_l1.T, del3.T)
            adjustment1 = dot(training_set_inputs.T, del2.T)
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3
    # The neural network thinks.
    def think(self, inputs,w):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, w))

    def res(self,inputs):
        hidden_l1 = self.think(inputs,self.synaptic_weights1)
        hidden_l2 = self.think(hidden_l1,self.synaptic_weights2)
        output = self.think(hidden_l2,self.synaptic_weights3)
        return output
if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()
    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights1
   # print neural_network.synaptic_weights1.shape
    print neural_network.synaptic_weights2
   # print neural_network.synaptic_weights2.shape
    print neural_network.synaptic_weights3
    #print neural_network.synaptic_weights3.shape
    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
    #print training_set_inputs.shape
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights1
    print neural_network.synaptic_weights2
    print neural_network.synaptic_weights3

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.res(array([1, 0, 0]))
