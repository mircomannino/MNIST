import matplotlib.pyplot as plt
import numpy as np
import time

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def update_line(g, x, y, ax):
    g.set_xdata(np.append(g.get_xdata(), x))
    g.set_ydata(np.append(g.get_ydata(), y))
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.1)


np.random.seed(int(time.time()))
class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        # self.weight = np.random.rand(self.weightDim + 1) * 0.01
        self.weight = np.random.uniform(low=-1, high=1, size=(self.weightDim + 1)) * 1
        self.gradient = np.zeros(self.weightDim + 1)
        self.delta_error = 0
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
    def printInfo(self):
        print("Gradient: ", self.gradient)
        print("Weight: ", self.weight)
        print("delta_error: ", self.delta_error)

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
                print("\t\t", self.layers[i].neurons[j].weight)
            print("-----------------")

    def train(self, train_file, learning_rate, n_epochs, scale_factor=1, plot=True):
        # Preparing the plot
        axes = plt.gca()
        axes.set_autoscale_on(True)
        #label of axes
        plt.ylabel('empirical risk')
        plt.xlabel('no epoch')
        g, = plt.plot([], [])

        # Get the data
        train_data = np.loadtxt(train_file, delimiter=",")
        train_labels = train_data[:, :1]
        train_imgs = train_data[:, 1:]

        nExamples = train_imgs.shape[0]
        miniBatchSize = 4

        for epoch in range(n_epochs):
            error = 0

            for k in range(nExamples):
                ###################### FORWARD STEP ############################
                X = []
                X.append(train_imgs[k])
                train_label_oneHot = (train_labels[k])
                nextInput = train_imgs[k]
                for layer in self.layers:
                    result = layer.calculateOutput(nextInput)
                    nextInput = result
                    X.append(result)
                prediction = X[len(X)-1]    # The prediction is the last element of X

                # Error
                for m in range(len(prediction)):
                    error += (prediction[m] - train_label_oneHot[m])**2

                ###################### BACKWARD STEP ###########################
                # h: Index of the layer N, N-1, N-2, ..., 1, 0
                # h: Correspond also the previous output, stored in X[]
                for h_inv in range(len(self.layers)):
                    h = (len(self.layers) - 1) - h_inv
                    if(h_inv == 0):         # Output layer
                        for i in range(self.layers[h].dim()):
                            Y_ki = train_label_oneHot[i]
                            sig_ki = prediction[i]
                            delta_error_ki = (sig_ki * (1 - sig_ki)) * (sig_ki - Y_ki)
                            self.layers[h].neurons[i].delta_error = delta_error_ki
                            for j in range(len(self.layers[h].neurons[i].weight) - 1):      # "-1" due to bias term
                                X_kj = X[h][j]
                                grad_ij = X_kj * delta_error_ki
                                self.layers[h].neurons[i].gradient[j] += grad_ij
                            bias_index = len(self.layers[h].neurons[i].weight) - 1
                            self.layers[h].neurons[i].gradient[bias_index] += 1 * delta_error_ki
                    else:
                        for i in range(self.layers[h].dim()):
                            sig_ki = X[h+1][i]
                            sum = 0
                            for c in range(self.layers[h+1].dim()):
                                w_ci = self.layers[h+1].neurons[c].weight[i]
                                delta_kc = self.layers[h+1].neurons[c].delta_error
                                sum += w_ci * delta_kc
                            delta_error_ki = (sig_ki * (1 - sig_ki)) * sum
                            self.layers[h].neurons[i].delta_error = delta_error_ki
                            for j in range(len(self.layers[h].neurons[i].weight) - 1):      # "-1" due to bias term
                                X_kj = X[h][j]
                                grad_ij = X_kj * delta_error_ki
                                self.layers[h].neurons[i].gradient[j] += grad_ij
                            bias_index = len(self.layers[h].neurons[i].weight) - 1
                            self.layers[h].neurons[i].gradient[bias_index] += 1 * delta_error_ki

            # Updating the weigths
            for layer in self.layers:
                for i in range(layer.dim()):
                    for j in range(len(layer.neurons[i].weight)):
                        before = layer.neurons[i].weight[j]
                        layer.neurons[i].weight[j] -= learning_rate * layer.neurons[i].gradient[j]
                        # if before != layer.neurons[i].weight[j]:
                        #     print("Neurone ", i, "Weight ", j, "Aggiornato da ", before, " a ", layer.neurons[i].weight[j])
                        #     print("PErchè gradient = ", layer.neurons[i].gradient[j])
                        layer.neurons[i].gradient[j] = 0

            if(plot):
                update_line(g, epoch, np.sqrt(error/(nExamples)), axes)
                print("Epoch: ", epoch, "Error: ", np.sqrt(error/(nExamples)))
            else:
                print("Epoch: ", epoch, "Error: ", np.sqrt(error/(nExamples)))
        plt.show()


    def predict(self, input, label=None):
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

    ############################ Training ######################################
    learning_rate = 1.5
    n_epochs = 500
    print("Starting training...")
    nn.train("data/xor_train.csv", learning_rate, n_epochs, plot=True)
    print("Training complete!\n")
    ############################################################################

    ##################### Set the correct weight [1] ###########################
    # nn.layers[0].neurons[0].weight[0] = 20
    # nn.layers[0].neurons[0].weight[1] = 20
    # nn.layers[0].neurons[0].weight[2] = -10
    # nn.layers[0].neurons[1].weight[0] = -20
    # nn.layers[0].neurons[1].weight[1] = -20
    # nn.layers[0].neurons[1].weight[2] = 30
    # nn.layers[1].neurons[0].weight[0] = 20
    # nn.layers[1].neurons[0].weight[1] = 20
    # nn.layers[1].neurons[0].weight[2] = -30
    ############################################################################

    ##################### Set the correct weight [1] ###########################
    # nn.layers[0].neurons[0].weight[0] = -7.05624541
    # nn.layers[0].neurons[0].weight[1] = 7.07011677
    # nn.layers[0].neurons[0].weight[2] = 3.59994854
    # nn.layers[0].neurons[1].weight[0] = -6.18799956
    # nn.layers[0].neurons[1].weight[1] = 5.93663609
    # nn.layers[0].neurons[1].weight[2] = -3.16858404
    # nn.layers[1].neurons[0].weight[0] = -10.53429254
    # nn.layers[1].neurons[0].weight[1] = 11.05323456
    # nn.layers[1].neurons[0].weight[2] = 4.99350977
    ############################################################################

    # Check the weights of each layers
    print("Weights before the prediction:")
    nn.checkLayers()
    print()

    ############################# Prediction ###################################
    print("XOR prediction")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for input in inputs:
        prediction = nn.predict(input)
        if(float(prediction[0]) > 0.5):
            print("Input: ", input, " Result: 1", "~", prediction)
        else:
            print("Input: ", input, " Result: 0", "~", prediction)
    ############################################################################

if __name__ == '__main__':
    main()
