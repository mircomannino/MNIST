import numpy as np
import time

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def one_hot_encoding(n):
    all_digit = np.arange(10)
    result = ((all_digit == n).astype(np.int))
    return result

def one_hot_decoding(n):
    print("One Hot DECODING")

class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        np.random.seed(int(time.time()))
        self.weight = np.random.rand(self.weightDim + 1) * 0.001
    def calculateOutput(self, input):
        output = .0
        if(self.weightDim != len(input)):
            return -1
        for i in range(self.weightDim):
            output += self.weight[i] * input[i]
        output += self.weight[self.weightDim]
        output = sigmoid(output)
        return output
    ### Test ###
    # myNeuron = Neuron(3);
    # input = [2, 2, 2]
    # print(myNeuron.calculateOutput(input))

class Layer:
    def __init__(self, layerDim, weightDim):
        self.neurons = []
        for i in range(layerDim):
            self.neurons.append(Neuron(weightDim))
    def calculateOutput(self, input):
        result = []
        for neuron in self.neurons:
            result.append(neuron.calculateOutput(input))
        return result

class NeuralNetwork:
    def __init__(self, nHiddenLayers, nHiddenUnits):
        self.nHiddenLayers = nHiddenLayers
        self.nHiddenUnits = nHiddenUnits
        # Create the first layer
        self.firstLayer = Layer(10, (28*28))
        # Create the hidden layers
        self.layers = []
        for i in range(nHiddenLayers):
            self.layers.append(Layer(nHiddenUnits, 10))
        # Create the output Layers
        self.outputLayer = Layer(10, 10)

    def train(self, train_file):
        # Get the train_data
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=2)
        train_labels = train_data[:, :1]
        train_imgs = train_data[:, 1:]
        nExamples = train_labels.shape[0]
        print("No. Examples: ", nExamples)
        ################
        # Forward step #
        ################
        for k in range(nExamples):
            train_label_oneHot = one_hot_encoding(train_labels[k])      # Target - OneHot
            inputHidden = self.firstLayer.calculateOutput(image)        # First layer
            tmp_input = inputHidden
            outputHidden = inputHidden
            for layer in self.layers:
                outputHidden = layer.calculateOutput(tmp_input)         # Hidden layer
                tmp_input = outputHidden
            prediction = self.outputLayer.calculateOutput(outputHidden) # Output layer


    def predict(self, test_file):
        # Get the test_data
        test_data = np.loadtxt(test_file, delimiter=",", max_rows=2)
        test_labels = test_data[:, :1]
        test_imgs = test_data[:, 1:]
        for image in test_imgs:
            # Calculates result of the firstLayer
            inputHidden = self.firstLayer.calculateOutput(image)
            # Propagation in the hidden layer
            tmp_input = inputHidden
            outputHidden = inputHidden
            for layer in self.layers:
                outputHidden = layer.calculateOutput(tmp_input)
                tmp_input = outputHidden
            # Calculates result of the output
            prediction = self.outputLayer.calculateOutput(outputHidden)
            print(prediction)

myNN = NeuralNetwork(2, 10)
myNN.train("data/mnist_train.csv")
myNN.predict("data/mnist_test.csv")
