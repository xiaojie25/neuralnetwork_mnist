import numpy as np
import scipy.special

# neural network class definition

class NeuralNetwork(object):

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) -> None:
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weight inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert input list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate sigmoid into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate sigmoid into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # callculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_error, split by weight, recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_errors)

        # update the weight for the linkd between the input and hidden layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signald into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signald emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signald into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals energing from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == 'main':
    pass