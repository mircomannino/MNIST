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
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=2)
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

                ##### Backward Step #####
                delta_errors = []               # Contains all the delta errors

                ### OUTPUT LAYER
                # i: Number of neuron
                # j: Number of weight of thw i-th neuron
                # h: index of X[], correspond to the previous layer of outputLayer
                h = len(X) - 2
                delta_error_output = []
                for i in range(self.outputLayer.dim()):
                    Y_ki = train_label_oneHot[i]
                    sig_ki = X[h+1][i]
                    delta_error_ki = (sig_ki * (1 - sig_ki)) * (sig_ki - Y_ki)
                    delta_error_output.append(delta_error_ki)
                    for j in range(len(X[h])):
                        X_kj = X[h][j]
                        grad_ij = delta_error_ki * X_kj
                        self.outputLayer.neurons[i].weight[j] -= learning_rate * grad_ij
                delta_errors.append(delta_error_output)

                ### HIDDEN LAYER

            print("Epoch: ", epoch, " Error: ", np.sqrt(error/(nExamples*10*2)))

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
