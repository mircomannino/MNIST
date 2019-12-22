import numpy as np
import time

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def one_hot_encoding(n):
    all_digit = np.arange(10)
    result = ((all_digit == n).astype(np.int))
    return result

np.random.seed(int(time.time()))
class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        self.weight = np.random.rand(self.weightDim + 1)  * 0.01
    def calculateOutput(self, input):
        output = 0.0
        if(self.weightDim != len(input)):
            return -1
            print("Erroree!")
        for i in range(self.weightDim):
            output += self.weight[i] * input[i]
        output += self.weight[self.weightDim]
        return sigmoid(output)
    def nWeight(self):
        return (self.weightDim + 1) # "+1" Due to the bias term

class Layer:
    def __init__(self, layerDim, weightDim):
        self.layerDim = layerDim
        self.neurons = []
        for i in range(layerDim):
            self.neurons.append(Neuron(weightDim))
    def calculateOutput(self, input):
        result = []
        for neuron in self.neurons:
            result.append(neuron.calculateOutput(input))
        return result
    def dim(self):
        return self.layerDim

class NeuralNetwork:
    def __init__(self, dimInput, nHidden, dimHiddenLayers, dimOutputLayer):
        self.dimInput = dimInput
        self.nHidden = nHidden
        self.dimHiddenLayers = dimHiddenLayers
        self.dimOutputLayer = dimOutputLayer
        self.layers = []
        for i in range(nHidden):
            if(i == 0):
                self.layers.append(Layer(dimHiddenLayers[i], dimInput))                 # First hidden layer
            else:
                self.layers.append(Layer(dimHiddenLayers[i], self.layers[i-1].dim()))   # Other hidden layers
        self.layers.append(Layer(dimOutputLayer, self.layers[nHidden-1].dim()))         # Output layer

    def checkLayers(self):
        for i in range(len(self.layers)):
            print("Layer ", i, ":")
            print("\t# Neurons: ", self.layers[i].dim())
            for j in range(len(self.layers[i].neurons)):
                print("\t\t# Weigth(", j, "): ", self.layers[i].neurons[j].nWeight())
            print("-----------------")

    def train(self, train_file, learning_rate, n_epochs):
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=100)
        train_labels = train_data[:, :1]
        train_imgs = train_data[:, 1:] / 255
        
        # Do the training...

    def predict(self, input, label):
        if len(input) != self.dimInput:
            return -1
        # FORWARD STEP
        nextInput = input
        for layer in self.layers:
            result = layer.calculateOutput(nextInput)
            nextInput = result
        return result

def main():
    dimInput = 28 * 28
    nHidden = 2
    dimHiddenLayers = [10, 10]
    dimOutputLayer = 10
    nn = NeuralNetwork(dimInput, nHidden, dimHiddenLayers, dimOutputLayer)
    #nn.checkLayers()

    # Train

    # Prediction
    test_file = "data/mnist_test.csv"
    test_data = np.loadtxt(test_file, delimiter=",", max_rows=2)
    test_labels = test_data[:, :1]
    test_imgs = test_data[:, 1:] / 255
    for k in range(len(test_imgs)):
        print(nn.predict(test_imgs[k], test_labels[k]))
    input_zero = np.zeros(784)
    print(nn.predict(input_zero, 0))


if __name__ == '__main__':
    main()
