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
    weight_range = 0.01
    W = np.random.rand(m, d+1) * weight_range       # No. of parameters
    error = np.zeros(m)                             # Error for each neuron
    learning_rate = 9e-1                            # Learning rate
    n_epochs = 200                                  # No. of epochs

    ##############################################
    # Read the train_data csv file #
    ##############################################
    # Load the train set
    train_data = np.loadtxt('data/mnist_train.csv', delimiter=',') # Get all the 60.000 examples
    l = train_data.shape[0]                                        # No. of examples
    A = np.zeros((l, m))                            # Argument of sigmoid function
    X = np.zeros((l, m))                            # Activation function
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
    mini_batch_size = 100 # (60000 / 200)
    for epoch in range(n_epochs):
        # Initialize the gradient and risk for each epoch
        W_grad = np.zeros((m, d+1))
        risk = 0
        for k_old in range(mini_batch_size):
            loss = 0
            for i in range(m):
                k = ((mini_batch_size * epoch) + k_old) % n_epochs
                A[k, i] = 0
                for j in range(d):
                    A[k, i] = A[k, i] +  W[i, j] * train_imgs[k, j]
                X[k, i] = 1/(1+math.exp(-A[k,i] - W[i,d]))
                # Calculate the loss (Relative entropy loss function)
                if(train_labels_one_hot[k, i] == 1):
                    loss = (-1) * train_labels_one_hot[k,i] * math.log(X[k, i])
                else:
                    loss = (-1) * (1 - train_labels_one_hot[k,i]) * math.log(1 - X[k,i])
                #print("Epoch: ", epoch, " Neuron: ", i, "Correct: ", train_labels_one_hot[k, i], " Predicted: ", X[k, i], " Loss: ", loss)
                risk += loss
                # Calculate the gradient
                if(train_labels_one_hot[k, i] == 1):
                    W_grad[i, ] = W_grad[i, ] - (1/mini_batch_size) * train_labels_one_hot[k,i] * (1 - X[k,i]) * train_imgs[k,]
                else:
                    W_grad[i, ] = W_grad[i, ] + (1/mini_batch_size) * (1 - train_labels_one_hot[k,i]) * X[k,i] * train_imgs[k,]
            risk += loss
        W -= (learning_rate/mini_batch_size) * W_grad
        print("Epoch: ", epoch, "   - Normalized error: ", (risk/(mini_batch_size * m)))

    ##########################
    # Save the trained model #
    ##########################
    file_name = 'trained_weights-bs' + str(mini_batch_size) + '.txt'
    np.savetxt(file_name, W, delimiter=',')


if __name__ == '__main__':
    main()
