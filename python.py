from net import *
import matplotlib.pyplot as plt


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 300
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

# create instance of neural network
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnis training data CSV file into a list
training_data_file = open("mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# go through all records in the training data set
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

