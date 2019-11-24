################################################################################
# MNIST - prediction of handwritten characters                                 #
# Mirco Mannino - 2019                                                         #
################################################################################
import numpy as np
import csv
import math
import sys
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


    ##############################
    # Prediction of the test set #
    ##############################
    print("\n\n############ Test set ############")
    if(sys.argv[1] == 'HLT'):   # High and Low thresold
        LThreshold = 0.49
        HThreshold = 0.51
        nError = 0
        for test_el in range(n):
            prediction = np.zeros(m)
            for i in range(m):
                for j in range(d):
                    prediction[i] += W[i, j] * test_imgs[test_el, j]
                prediction[i] = sigmoid(prediction[i] + W[i, d])
            if((test_data[test_el, i] == 0 and prediction[i] > LThreshold) or (test_data[test_el, i] == 1 and prediction[i] < HThreshold)):
                nError += 1
                i = m
                print("No. Test: ", test_el, "-> Wrong")
            print("No. Test: ", test_el, "-> Correct")
        print("Correct percentage: ", ((n - nError)/n)*100, "%")
    elif(sys.argv[1] == 'MAX2'):    # View the two higher/lower value;
        # TODO: Implementa con verificando se la differenza tra i due valori più
        #       alti supera una certa soglia
        print("Ciao mirco")

if __name__ == '__main__':
    main()
