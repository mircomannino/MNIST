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
        self.weight = np.random.rand(self.weightDim + 1)  * 0.1
    def calculateOutput(self, input):
        output = 0.0
        if(self.weightDim != len(input)):
            return -1
            print("Erroree!")
        for i in range(self.weightDim):
            output += self.weight[i] * input[i]
        output += self.weight[self.weightDim]
        return sigmoid(output)

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
        for layer in self.layers:
            print("Layer:")
            print(layer.dim())
            for neuron in layer.neurons:
                print(neuron.weight)
            print("-----------------")

    def predict(self, input):
        if len(input) != self.dimInput:
            return -1
        # FORWARD STEP
        nextInput = input
        for layer in self.layers:
            result = layer.calculateOutput(nextInput)
            nextInput = result
        return result

def main():
    dimInput = 2
    nHidden = 1
    dimHiddenLayers = [2]
    dimOutputLayer = 1
    nn = NeuralNetwork(dimInput, nHidden, dimHiddenLayers, dimOutputLayer)
    # Set the correct weight
    nn.layers[0].neurons[0].weight[0] = 20
    nn.layers[0].neurons[0].weight[1] = 20
    nn.layers[0].neurons[0].weight[2] = -10
    nn.layers[0].neurons[1].weight[0] = -20
    nn.layers[0].neurons[1].weight[1] = -20
    nn.layers[0].neurons[1].weight[2] = 30
    nn.layers[1].neurons[0].weight[0] = 20
    nn.layers[1].neurons[0].weight[1] = 20
    nn.layers[1].neurons[0].weight[2] = -30
    # Prediction
    print("XOR prediction")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for input in inputs:
        prediction = nn.predict(input)
        if(float(prediction[0]) > 0.5):
            print("Input: ", input, " Result: 1", "~", prediction)
        else:
            print("Input: ", input, " Result: 0", "~", prediction)

if __name__ == '__main__':
    main()
