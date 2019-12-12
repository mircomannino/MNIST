import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def one_hot_encoding(n)
    print("One Hot ENCODING")

def one_hot_decoding(n):
    print("One Hot DECODING")

class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        self.weight = np.random.rand(self.weightDim + 1) * 0.01
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
    def __init__(self, nHiddenLayers, nHiddenUnits, train_file):
        self.nHiddenLayers = nHiddenLayers
        self.nHiddenUnits = nHiddenUnits
        # Get the train_data
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=2)
        self.train_labels = train_data[:, :1]
        self.train_imgs = train_data[:, 1:]
        # Create the first layer
        self.firstLayer = Layer(10, (28*28))
        # Create the hidden layers
        self.layers = []
        for i in range(nHiddenLayers):
            self.layers.append(Layer(nHiddenUnits, 10))
        # Create the output Layers
        self.outputLayer = Layer(10, 10)

    def INFO(self):
        print("Info about your network:")
        print("No. Hidden layers:\t", len(self.layers))
        print("No. Hidden units:\t", len(self.layers[0].neurons))
        print("No. Output units:\t", len(self.outputLayer.neurons))

    def train(self):
        print("Train...")

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
            outputHidden = []
            for layer in self.layers:
                outputHidden = layer.calculateOutput(tmp_input)
                tmp_input = outputHidden
            # Calculates result of the output
            result = self.outputLayer.calculateOutput(outputHidden)
            print(result)

myNN = NeuralNetwork(5, 10, "data/mnist_train.csv")
myNN.INFO()
myNN.predict("data/mnist_test.csv")
