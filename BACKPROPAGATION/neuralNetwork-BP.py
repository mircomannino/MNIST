import matplotlib.pyplot as plt
import numpy as np
import time

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def one_hot_encoding(n):
    all_digit = np.arange(10)
    result = ((all_digit == n).astype(np.int))
    return result

def update_line(g, x, y, ax):
    g.set_xdata(np.append(g.get_xdata(), x))
    g.set_ydata(np.append(g.get_ydata(), y))
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.1)

def argmax(list):
    max_index = 0
    max = 0
    for i in range(len(list)):
        if (list[i] > max):
            max = list[i]
            max_index = i
    return max_index

np.random.seed(int(time.time()))
class Neuron:
    def __init__(self, weightDim):
        self.weightDim = weightDim
        #self.weight = np.random.rand(self.weightDim + 1) * 0.1
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
        if nHidden > 0:                                                         # Output layer
            self.layers.append(Layer(dimOutputLayer, self.layers[nHidden-1].dim()))
        else:
            self.layers.append(Layer(dimOutputLayer, dimInput))

    def checkLayers(self):
        for i in range(len(self.layers)):
            print("Layer ", i, ":")
            print("\t# Neurons: ", self.layers[i].dim())
            for j in range(len(self.layers[i].neurons)):
                print("\t\t# Weigth(", j, "): ", self.layers[i].neurons[j].nWeight())
                print("\t\t", self.layers[i].neurons[j].weight)
            print("-----------------")

    def train(self, train_file, learning_rate, n_epochs, scale_factor=1, plot=False):
        # Preparing the plot
        axes = plt.gca()
        axes.set_autoscale_on(True)
        #label of axes
        plt.ylabel('quadratic-error')
        plt.xlabel('n. epoch')
        g, = plt.plot([], [])

        # Get the data
        train_data = np.loadtxt(train_file, delimiter=",", max_rows=10000)
        train_labels = train_data[:, :1]
        train_imgs = train_data[:, 1:] / scale_factor

        nExamples = train_imgs.shape[0]
        miniBatchSize = 100

        for epoch in range(n_epochs):
            error = 0

            for k_tot in range(miniBatchSize):
                ###################### FORWARD STEP ############################
                k = ((epoch * miniBatchSize) + k_tot) % nExamples
                X = []
                X.append(train_imgs[k])
                train_label_oneHot = one_hot_encoding(train_labels[k])
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
        if(plot):
            plt.show()

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

    all_learning_rate = [0.06]

    for learning_rate in all_learning_rate:

        dimInput = 28 * 28
        nHidden = 1
        dimHiddenLayers = [30]
        dimOutputLayer = 10
        nn = NeuralNetwork(dimInput, nHidden, dimHiddenLayers, dimOutputLayer)
        #nn.checkLayers()


        # # Prediction
        # test_file = "data/mnist_test.csv"
        # test_data = np.loadtxt(test_file, delimiter=",", max_rows=5)
        # test_labels = test_data[:, :1]
        # test_imgs = test_data[:, 1:] / 255
        # tot_pred = test_data.shape[0]
        # correct_pred = 0
        # for k in range(len(test_imgs)):
        #     prediction = nn.predict(test_imgs[k], test_labels[k])
        #     print(test_labels[k], prediction)
        #     my_prediction = int(argmax(prediction))
        #     real_prediction = int(test_labels[k])
        #     print("Ho predetto: ", my_prediction, " Corretto: ", real_prediction)
        #     if (my_prediction == real_prediction):
        #         correct_pred += 1
        # print("Total: ", tot_pred)
        # print("Correct: ", correct_pred)


        # Train
        train_file = "data/mnist_train.csv"
        # learning_rate = 0.1
        n_epochs = 300
        scale_factor = 255
        nn.train(train_file, learning_rate, n_epochs, scale_factor, False)

        # INfo output layer
        outIndex = len(nn.layers) - 1
        for i in range(len(nn.layers[outIndex].neurons)):
            print("Neuron ", i)
            print(nn.layers[outIndex].neurons[i].delta_error)

        # Prediction
        test_file = "data/mnist_test.csv"
        test_data = np.loadtxt(test_file, delimiter=",", max_rows=1000)
        test_labels = test_data[:, :1]
        test_imgs = test_data[:, 1:] / 255
        tot_pred = test_data.shape[0]
        correct_pred = 0
        for k in range(len(test_imgs)):
            prediction = nn.predict(test_imgs[k], test_labels[k])
            print(test_labels[k], prediction)
            my_prediction = int(argmax(prediction))
            real_prediction = int(test_labels[k])
            print("Ho predetto: ", my_prediction, " Corretto: ", real_prediction)
            if (my_prediction == real_prediction):
                correct_pred += 1
        print("Total: ", tot_pred)
        print("Correct: ", correct_pred)

        accuracy = (correct_pred * 100) / tot_pred
        with open("result-MNIST.txt", "a+") as out_file:
            out_msg = "learning_rate: " + str(learning_rate) + "\t accuracy: " + str(accuracy) + "%\n"
            out_file.write(out_msg)


if __name__ == '__main__':
    main()
