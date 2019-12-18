################################################################################
# MNIST - prediction of handwritten characters                                 #
# Mirco Mannino - 2019                                                         #
################################################################################
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

def show_images(dataset_csv, n_images):
    with open(dataset_csv) as dataset:
        counter = 0
        for data in csv.reader(dataset):
            # Take the label
            label = data[0]
            # Take and reshape the pixels
            pixels_1D = data[1:]
            pixels_28x28 = (np.array(pixels_1D, dtype='uint8')).reshape(28, 28)
            # Show the images
            plt.title('Label: {label}'.format(label=label))
            plt.imshow(pixels_28x28, cmap='gray')
            plt.show()
            # Show only the {n_images} images
            counter += 1
            if counter >= n_images:
                break

def sigmoid(x):
    return float((1 / (1 + np.exp(-x))))

def one_hot_encoding(decoded_array):
    encoded_array = np.zeros(len(decoded_array))
    max_value = 0
    max_index = 0
    for i in range(len(decoded_array)):
        if decoded_array[i] > max_value:
            max_value = decoded_array[i]
            max_index = i
    encoded_array[max_index] = 1
    return encoded_array

def one_hot_to_digit(one_hot):
    for i in range(one_hot.shape[0]):
        if one_hot[i] == 1:
            return i


def main():
    # ## Show the first 5 images ###
    # show_images('data/mnist_train.csv', 5)

    #############################
    # Initialize the parameters #
    #############################
    np.random.seed(42)
    m = 10                                          # No. of neurons
    d = 28*28                                       # No. of pixels for each image
    l = 50                                          # No. of examples
    weight_range = 0.01
    W = np.random.rand(m, d+1) * weight_range       # No. of parameters
    A = np.zeros((l, m))                            # Argument of sigmoid function
    X = np.zeros((l, m))                            # Activation function
    error = np.zeros(m)                             # Error for each neuron
    learning_rate = 9e-1                            # Learning rate
    n_epochs = 250                                  # No. of epochs

    ##############################################
    # Read the train_data csv file #
    ##############################################
    # Load the train set
    train_data = np.loadtxt('data/mnist_train.csv', delimiter=',', max_rows=l)
    train_labels = train_data[:, :1]
    train_data = train_data[:, 1:]
    # Normalize the training image
    train_imgs = np.ones((l, d+1))
    train_factor = 1/255
    for k in range(l):
        for j in range(d):
            train_imgs[k, j] = train_data[k, j] * train_factor

    ####################
    # One-Hot encoding #
    ####################
    # View the general encoding
    all_digits = np.arange(10)
    print("\nONE-HOT encoding")
    for label in range(10):
        one_hot = (all_digits == label).astype(np.int)
        print(label, " to one-hot: ", one_hot)
    # Encoding the labels
    LowValue=0
    HighValue=1
    train_labels_one_hot = (all_digits == train_labels).astype(np.int)
    train_labels_one_hot[train_labels_one_hot == 0] = LowValue
    train_labels_one_hot[train_labels_one_hot == 1] = HighValue
    #
    #   Note the train_labels_one_hot[train_labels_one_hot==0] return the
    #   elements of the array that are equal to zero
    #

    #######################
    # Print all the shape #
    #######################
    print("\nShape of the parameters")
    print("W:\t\t", W.shape)
    print("A\t\t", A.shape)
    print("X:\t\t", X.shape)
    print("error:\t\t", error.shape)
    print("train_imgs:\t", train_imgs.shape)
    print("train_labels:\t", train_labels.shape)
    print("train_labels_one_hot\t", train_labels_one_hot.shape)



    #############################
    # Training of the alogrithm #
    #############################
    for epoch in range(n_epochs):
        # Initialize the gradient and risk for each epoch
        W_grad = np.zeros((m, d+1))
        risk = 0
        for k in range(l):
            loss = 0
            prediction = []
            for i in range(m):
                A[k, i] = 0
                for j in range(d):
                    A[k, i] = A[k, i] +  W[i, j] * train_imgs[k, j]
                X[k, i] = sigmoid(A[k, i] - W[i, d]) # 1/(1+math.exp(-A[k,i] - W[i,d]))   #
                prediction.append(X[k, i])
                loss += (X[k, i] - train_labels_one_hot[k, i])**2
                error[i] =  X[k, i] - train_labels_one_hot[k, i]
                W_grad[i, ] += error[i] * X[k, i] * (1 - X[k, i]) * train_imgs[k, ]
            risk += loss
            print("prediction: ", prediction)
        W -= (learning_rate/l) * W_grad
        print("Epoch: ", epoch, "   - Normalized error: ", math.sqrt(risk/(l * m)))

    ##########################
    # Save the trained model #
    ##########################
    np.savetxt('trained_weights.txt', W, delimiter=',')


if __name__ == '__main__':
    main()
