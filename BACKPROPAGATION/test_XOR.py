import numpy as np
import time

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def one_hot_encoding(n):
    all_digit = np.arange(10)
    result = ((all_digit == n).astype(np.int))
    return result

#np.random.seed(int(time.time()))
np.random.seed(1024)
class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        self.weight = np.random.rand(self.weightDim + 1) * 10
    def calculateOutput(self, input):
        output = 0.0
        if(self.weightDim != len(input)):
            return -1
        for i in range(self.weightDim):
            output += self.weight[i] * input[i]
        output += self.weight[self.weightDim]
        return sigmoid(output)
    ### Test ###
    # myNeuron = Neuron(3);
    # input = [2, 2, 2]
    # print(myNeuron.calculateOutput(input))

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
    def printWeight(self):
        for neuron in self.neurons:
            print(neuron.weight)

### TEST ###
hiddenLayer = Layer(2, 2)
outputLayer = Layer(1, 2)

input = [ [0, 0], [0, 1], [1, 0], [1, 1] ]

print("Hidden prima:")
hiddenLayer.printWeight()
hiddenLayer.neurons[0].weight[0] = 20
hiddenLayer.neurons[0].weight[1] = 20
hiddenLayer.neurons[0].weight[2] = -10
hiddenLayer.neurons[1].weight[0] = -20
hiddenLayer.neurons[1].weight[1] = -20
hiddenLayer.neurons[1].weight[2] = 30
print("Hidden dopo:")
hiddenLayer.printWeight()


res_hidden = hiddenLayer.calculateOutput(input[0])
print(res_hidden)

print("Output prima:")
outputLayer.printWeight()

outputLayer.neurons[0].weight[0] = 20
outputLayer.neurons[0].weight[1] = 20
outputLayer.neurons[0].weight[2] = -30
print("Output dopo:")
outputLayer.printWeight()

res_out = outputLayer.calculateOutput(res_hidden)
print("Raw result: ", res_out)
if (float(res_out[0]) > 0.5):
    print("Final result: 1")
else:
    print("Final result: 0")
