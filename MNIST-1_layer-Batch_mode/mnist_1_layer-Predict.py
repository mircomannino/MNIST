################################################################################
# MNIST - prediction of handwritten characters                                 #
# Mirco Mannino - 2019                                                         #
################################################################################
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

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
    #############################
    # Initialize the parameters #
    #############################
    np.random.seed(42)
    m = 10                                                      # No. of neurons
    d = 28*28                                                   # No. of pixels for each image
    l = 50                                                      # No. of examples
    W = W = np.loadtxt('trained_weights.txt', delimiter=',')    # Trained weights

    ##############################################
    # Read the test_data csv file #
    ##############################################
    # Load the test set
    test_data  = np.loadtxt('data/mnist_test.csv',  delimiter=',')
    test_labels = test_data[:, :1]
    test_data = test_data[:, 1:]
    # Normalize the test image
    n = test_data.shape[0]                         # No. of test item
    test_factor = 1/255
    test_imgs = np.ones((n, d+1))
    for k in range(n):
        for j in range(d):
            test_imgs[k, j] = test_data[k, j] * test_factor

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
    test_labels_one_hot = (all_digits == test_labels).astype(np.int)
    test_labels_one_hot[test_labels_one_hot == 0] = LowValue
    test_labels_one_hot[test_labels_one_hot == 1] = HighValue

    ##############################
    # Prediction of the test set #
    ##############################
    print("\n\n############ Test set ############")
    print("No.\tCorrect\tPredicted")
    corrects = 0
    for test_el in range(n):
        # TODO: make prediction and test if it's correct
        # Visualize the correct result
        #   print(test_labels_one_hot[test_el], " -> ", one_hot_to_digit(test_labels_one_hot[test_el]))
        prediction = np.zeros(m)
        for i in range(m):
            for j in range(d):
                prediction[i] += W[i, j] * test_imgs[test_el, j]
            prediction[i] = sigmoid(prediction[i])
        correct_answer = one_hot_to_digit(test_labels_one_hot[test_el])
        all_digits = np.arange(m)
        prediction_one_hot = one_hot_encoding(prediction)
        prediction_digit = one_hot_to_digit(prediction_one_hot)
        # Check the correctness
        if prediction_digit == correct_answer:
            corrects += 1
        print(test_el, '\t', correct_answer, "\t", prediction_digit)
    print("Correct percentage: ", ((corrects/n)*100), "%")

if __name__ == '__main__':
    main()
