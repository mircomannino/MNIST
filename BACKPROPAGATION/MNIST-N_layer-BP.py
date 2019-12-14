import numpy as np
import time

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_prime(a):
    return ( sigmoid(a) * (1 - sigmoid(a)) )

def one_hot_encoding(n):
    all_digit = np.arange(10)
    result = ((all_digit == n).astype(np.int))
    return result

def one_hot_decoding(n):
    print("One Hot DECODING")

np.random.seed(int(time.time()))
class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        self.weight = np.random.rand(self.weightDim + 1) * 0.01
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
    ### Test ###
    # print("Test layer")
    # myL1 = Layer(5, 3)
    # input1 = [12./255, 100./255, 0./255]
    # input2 = [200./255., 123./255., 0]
    # # print("Case1: ", myL1.calculateOutput(input1))
    # # print("Case2: ", myL1.calculateOutput(input2))
    # myL2 = Layer(2, 5)
    # print("Case1: ", myL2.calculateOutput(myL1.calculateOutput(input1)))
    # print("Case2: ", myL2.calculateOutput(myL1.calculateOutput(input2)))

class NeuralNetwork:
    def __init__(self, nHiddenLayers, nHiddenUnits):
        self.nHiddenLayers = nHiddenLayers
        self.nHiddenUnits = nHiddenUnits
        # Create the first layer
        self.firstLayer = Layer(10, (28*28))
        # Create the hidden layers
        self.hiddenLayers = []
        for i in range(nHiddenLayers-1):
            self.hiddenLayers.append(Layer(nHiddenUnits, 10))
        # Create the output Layers
        self.outputLayer = Layer(10, 10)

    def train(self, train_file):
        # Get the train_data
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=100)
        train_labels = train_data[:, :1]
        train_imgs = train_data[:, 1:] / 255.0
        nExamples = train_labels.shape[0]
        print("No. Examples: ", nExamples)
        n_epochs = 100
        learning_rate = 0.1
        for epoch in range(n_epochs):
            error = 0
            for k in range(nExamples):
                ##### Forward Step #####
                X = []
                train_label_oneHot = one_hot_encoding(train_labels[k])              # Target - OneHot
                inputHidden = self.firstLayer.calculateOutput(train_imgs[k])        # First layer
                X.append(inputHidden)
                tmp_input = inputHidden
                outputHidden = inputHidden
                for layer in self.hiddenLayers:
                    outputHidden = layer.calculateOutput(tmp_input)                 # Hidden layer
                    X.append(outputHidden)
                    tmp_input = outputHidden
                prediction = self.outputLayer.calculateOutput(outputHidden)         # Output layer
                # Error
                for m in range(len(prediction)):
                    error += (1/2)*(prediction[m] - train_label_oneHot[m])**2
                #print(prediction)
                X.append(prediction)

                ##### Backward Step #####
                delta_errors = []
                ### Output layer
                # j: Number of Layer
                # i: Number of neuron
                j = len(X)-1
                delta_error_out = []
                for i in range(self.outputLayer.dim()):
                    x = X[j][i]
                    delta_error = (x * (1 - x) * (x - train_label_oneHot[i]))
                    delta_error_out.append(delta_error)
                    grad = delta_error * x
                    self.outputLayer.neurons[i].weight[i] -= learning_rate * grad
                delta_errors.append(delta_error_out)
                ### Hidden layer
                # j: Number of Layer  N, N-1, ..., 0
                # inv_j: Number of layer 0, 1, ..., N
                # i: Number of neuron
                # c: Number of children
                d_counter = 0   # Index for the delta_errors
                for inv_j in range(len(self.hiddenLayers)):
                    j = len(X) - (2 + inv_j)
                    delta_error_hidden = []
                    for i in range(self.hiddenLayers[j-1].dim()):
                        x = X[j-1][i]
                        sum = 0 # SUM(Wij * dkj) over j [ Nel quaderno ]
                        for c in range(10):
                            delta_error_children = delta_errors[d_counter][c]
                            h_weight = 0
                            if(inv_j == 0):
                                h_weight = self.outputLayer.neurons[i].weight[c]
                            else:
                                h_weight = self.hiddenLayers[j].neurons[i].weight[c]
                            sum += h_weight * delta_error_children
                        delta_error = x * sum
                        delta_error_hidden.append(delta_error)
                        grad = delta_error * x
                        self.hiddenLayers[j-1].neurons[i].weight -= learning_rate * grad
                    delta_errors.append(delta_error_hidden)
                    d_counter += 1
                ### First Layer
                # j: Number of Layer
                # i: Number of neuron
                for i in range(self.firstLayer.dim()):
                    x = X[0][i]
                    sum = 0 # SUM(Wij * dkj) over j [ Nel quaderno ]
                    for c in range(10):
                        delta_error_children = delta_errors[d_counter][c]
                        h_weight = self.hiddenLayers[0].neurons[i].weight[c]
                        sum += h_weight * delta_error_children
                    delta_error = x * sum
                    grad = delta_error * x
                    self.firstLayer.neurons[i].weight -= learning_rate * grad
            print("Epoch: ", epoch, " Error: ", np.sqrt(error/(nExamples*10)))

    def predict(self, test_file):
        # Get the test_data
        test_data = np.loadtxt(test_file, delimiter=",", max_rows=200)
        test_labels = test_data[:, :1]
        test_imgs = test_data[:, 1:] / 255.
        for image in test_imgs:
            # Calculates result of the firstLayer
            inputHidden = self.firstLayer.calculateOutput(image)
            # Propagation in the hidden layer
            tmp_input = inputHidden
            outputHidden = inputHidden
            for layer in self.hiddenLayers:
                outputHidden = layer.calculateOutput(tmp_input)
                tmp_input = outputHidden
            # Calculates result of the output
            prediction = self.outputLayer.calculateOutput(outputHidden)
            print("My-prediction: ", prediction)

myNN = NeuralNetwork(5, 10)
myNN.train("data/mnist_train.csv")
myNN.predict("data/mnist_test.csv")
