import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

weight1 = np.random.rand(785) * 0.1
weight2 = np.random.rand(785) * 0.1

data = np.loadtxt("data/mnist_train.csv", delimiter=",", max_rows=2)
input = data[:, 1:] / 255

print(len(input[0]))
sum1 = 0
sum2 = 0
for i in range(784):
    sum1 +=  input[0][i] * weight1[i]
    sum2 +=  input[0][i] * weight2[i]
sum1 += weight1[784]
sum2 += weight2[784]

print("SUM 1")
print(sum1)
print(sigmoid(sum1))

print("SUM 2")
print(sum2)
print(sigmoid(sum2))
