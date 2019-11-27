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

def max(n1, n2):
    if(n1 > n2):
        return n1
    else:
        return n2

def main():
    #############################
    # Initialize the parameters #
    #############################
    np.random.seed(42)
    m = 10                                                      # No. of neurons
    d = 28*28                                                   # No. of pixels for each image
    l = 50                                                      # No. of examples
    W = np.loadtxt(sys.argv[2], delimiter=',')                  # Trained weights

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
    # Convert the label in one_hot
    list = np.arange(10)
    test_labels_one_hot = (list==test_labels).astype(np.float)


    ##############################
    # Prediction of the test set #
    ##############################
    print("\n\n############ Test set ############")
    nError = 0
    #n = 10 ##################################################################### Da togliere
    if(sys.argv[1] == 'HLT'):   # High and Low thresold
        LThreshold = 0.49
        HThreshold = 0.51
        for test_el in range(n):
            raw_prediction = np.zeros(m)
            prediction = np.zeros(m)
            correct = True
            for i in range(m):
                for j in range(d):
                    raw_prediction[i] += W[i, j] * test_imgs[test_el, j]
                prediction[i] = sigmoid(raw_prediction[i] + W[i, d])
                if((test_labels_one_hot[test_el, i] == 0 and prediction[i] > LThreshold) or (test_labels_one_hot[test_el, i] == 1 and prediction[i] < HThreshold)):
                    nError += 1
                    correct = False
                    print("No. Test: ", test_el, "-> Wrong ")
                    break
            if(correct):
                print("No. Test: ", test_el, "-> Correct", "(", test_labels[test_el], ")")
        print("Total examples: ", n)
        print("Correct: ", n-nError)
        print("Wrong: ", nError)
        print("Correct percentage: ", ((n - nError)/n)*100, "%")
    elif(sys.argv[1] == 'MAX2'):    # View the two higher value;
        thresold = 0.1
        for test_el in range(n):
            raw_prediction = np.zeros(m)
            prediction = np.zeros(m)
            for i in range(m):
                for j in range(d):
                    raw_prediction[i] += W[i, j] * test_imgs[test_el, j]
                prediction[i] = sigmoid(raw_prediction[i] + W[i, d])
            #print(prediction)
            max1 = 0
            neuron1 = 0
            max2 = 0
            neuron2 = 0
            for i in range(len(prediction)):
                if prediction[i] > max1:
                    max1 = prediction[i]
                    neuron1 = i
            for i in range(len(prediction)):
                if (prediction[i] > max2) and (prediction[i] != max1):
                    max2 = prediction[i]
                    neuron2 = i
            my_prediction = 0
            if(abs(neuron1 - neuron2) > thresold):
                my_prediction = max(neuron1, neuron2)
            if(my_prediction == test_labels[test_el]):
                print("No. Test: ", test_el, "-> Correct", "(", test_labels[test_el], ")")
            else:
                print("No. Test: ", test_el, "-> Wrong")
                nError += 1
            # print(neuron1, " ", neuron2)
            # print(my_prediction, " ", test_labels[k])
            # print("Correct: ", test_labels[test_el])
            # print("Prediction: ", my_prediction)
            # print("Neurons: ", prediction)
            # print("Max1: ", max1)
            # print("Max2: ", max2)
            # print("#########################")
        print("Correct percentage: ", ((n - nError)/n)*100, "%")

if __name__ == '__main__':
    main()
