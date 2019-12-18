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
        self.weight = np.random.rand(self.weightDim + 1) * 0.1
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
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=2)
        train_labels = train_data[:, :1]
        train_imgs = train_data[:, 1:] / 255
        nExamples = train_labels.shape[0]
        print("No. Examples: ", nExamples)
        n_epochs = 100
        learning_rate = 0.7
        for epoch in range(n_epochs):
            error = 0
            for k in range(nExamples):
                ##### Forward Step #####
                train_label_oneHot = one_hot_encoding(train_labels[k])              # Target - OneHot
                X = []
                #           X = [   [input],
                #                   [result of HL1],
                #                   [result of HL2],
                #                   ...,
                #                   [result of the OL]
                #               ]
                X.append(train_imgs[k])
                firstLayerOutput = self.firstLayer.calculateOutput(train_imgs[k])   # First layer
                X.append(firstLayerOutput)
                tmp_input = firstLayerOutput
                outputHidden = firstLayerOutput
                for layer in self.hiddenLayers:
                    outputHidden = layer.calculateOutput(tmp_input)                 # Hidden layer
                    X.append(outputHidden)
                    tmp_input = outputHidden
                prediction = self.outputLayer.calculateOutput(outputHidden)         # Output layer
                X.append(prediction)
                #print(prediction)
                # Error
                for m in range(len(prediction)):
                    error += (prediction[m] - train_label_oneHot[m])**2

                ### DEBUG ###
                # for i in range(len(X)):
                #     if i != 0:
                #         print(X[i])
                # print("___________________________")
                #############

                ##### Backward Step #####
                delta_errors = []               # Contains all the delta errors

                ### OUTPUT LAYER
                # i: Number of neuron
                # j: Number of weight of the i-th neuron
                # h: index of X[], correspond to the previous layer of outputLayer
                h = len(X) - 2
                delta_error_output = []
                for i in range(self.outputLayer.dim()):
                    Y_ki = train_label_oneHot[i]
                    sig_ki = X[h+1][i]  # The prediction of the net
                    delta_error_ki = (sig_ki * (1 - sig_ki)) * (sig_ki - Y_ki)
                    delta_error_output.append(delta_error_ki)
                    for j in range(len(X[h])):
                        X_kj = X[h][j]
                        # print("OUT, Neuron ", i, " Weight ", j, "~> X_kj: ", X_kj, "--- sig(a_ki): ", sig_ki)
                        grad_ij = delta_error_ki * X_kj
                        self.outputLayer.neurons[i].weight[j] -= learning_rate * grad_ij
                        #print("grad_ij = ", grad_ij, " - ", i, j)
                delta_errors.append(delta_error_output)

                ### HIDDEN LAYER
                # i: Number of neuron
                # c: NUmber of children of the i-th neuron
                # j: NUmber of weight of the i-th neuron
                # inv_n_layer: Number of layer 0, 1, 2, 3... (The first is not count)
                # n_layer: Number of the current layer  = (len(self.hiddenLayers) - 1) - inv_n_layer
                # h: index of X[], correspond to the previous layer of the current layer
                delta_counter = 0 # Start from 1 because delta_errors[0] correspond to the delta errors of the output
                for inv_n_layer in range(len(self.hiddenLayers)):
                    n_layer = (len(self.hiddenLayers) - 1) - inv_n_layer
                    h = n_layer + 1
                    delta_error_hidden = []
                    for i in range(self.hiddenLayers[n_layer].dim()):
                        sig_ki = X[h+1][i]
                        delta_error_ki = 0
                        sum = 0
                        if (inv_n_layer == 0):        # Last hidden layer
                            for c in range(self.outputLayer.dim()):
                                W_ci = self.outputLayer.neurons[c].weight[i]
                                sum += delta_errors[0][c] * W_ci
                        else:                        # Other hidden layer
                            for c in range(self.hiddenLayers[n_layer+1].dim()):
                                W_ci = self.hiddenLayers[n_layer+1].neurons[c].weight[i]
                                sum += delta_errors[delta_counter][c] * W_ci
                        delta_error_ki = (sig_ki * (1 - sig_ki)) * sum
                        delta_error_hidden.append(delta_error_ki)
                        for j in range(len(X[h])):
                            X_kj = X[h][j]
                            grad_ij = X_kj * delta_error_ki
                            self.hiddenLayers[n_layer].neurons[i].weight[j] -= learning_rate * grad_ij
                    delta_errors.append(delta_error_hidden)
                    delta_counter += 1

                ### FIRST LAYER
                for i in range(self.firstLayer.dim()):
                    sig_ki = X[1][i]
                    sum = 0
                    for c in range(self.hiddenLayers[0].dim()):
                        W_ci = self.hiddenLayers[0].neurons[c].weight[i]
                        sum += delta_errors[len(delta_errors)-1][c] * W_ci
                    delta_error_ki = (sig_ki * (1 - sig_ki)) * sum
                    for j in range(len(X[0])):
                        X_kj = X[0][j]
                        grad_ij = X_kj * delta_error_ki
                        self.firstLayer.neurons[i].weight[j] -= learning_rate * grad_ij
                        # if(epoch == 1):
                        #     print("Grad", i, j, "~> ", grad_ij, "Beacuase: ", X_kj, " * ", delta_error_ki)

            print("Epoch: ", epoch, " Error: ", np.sqrt(error/(nExamples*10)))

    def printWeightOut(self):
        for neuron in self.outputLayer.neurons:
            print("Neuron:")
            print(neuron.weight)
            print()

    def printWeightFirst(self):
        print("Neuron 1:")
        print(self.firstLayer.neurons[0].weight)
        print()

    def predict(self, test_file):
        # Get the test_data
        test_data = np.loadtxt(test_file, delimiter=",", max_rows=3)
        test_labels = test_data[:, :1]
        test_imgs = test_data[:, 1:] / 255
        k = 0
        for image in test_imgs:
            # Calculates result of the firstLayer
            firstLayerOutput = self.firstLayer.calculateOutput(image)
            print("Input layer: ", firstLayerOutput)
            # Propagation in the hidden layer
            tmp_input = firstLayerOutput
            outputHidden = firstLayerOutput
            for layer in self.hiddenLayers:
                outputHidden = layer.calculateOutput(tmp_input)
                print("Hidden layer: ", outputHidden)
                tmp_input = outputHidden
            # Calculates result of the output
            prediction = self.outputLayer.calculateOutput(outputHidden)
            print(test_labels[k], "My-prediction: ", prediction)
            print("::::::::::::::::::::::::::::::::::")
            k += 1

### MNIST ###
myNN = NeuralNetwork(3, 10)
# myNN.printWeightOut()
myNN.train("data/mnist_train.csv")
# myNN.printWeightOut()
myNN.predict("data/mnist_test.csv")
