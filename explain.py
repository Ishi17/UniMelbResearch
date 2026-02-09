import numpy as np

#decide the size of each layer
n = [2, 3, 3, 1]
#Check the size
# print("layer 0 / input layer size", n[0])
# print("layer 1 size", n[1])
# print("layer 2 size", n[2])
# print("layer 3 size", n[3])

#creates a weight matrix and bias matrix for every layer with the shape given in the brackets
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)
#Check the shape
# print("Weights for layer 1 shape:", W1.shape)
# print("Weights for layer 2 shape:", W2.shape)
# print("Weights for layer 3 shape:", W3.shape)
# print("bias for layer 1 shape:", b1.shape)
# print("bias for layer 2 shape:", b2.shape)
# print("bias for layer 3 shape:", b3.shape)

#Input data shape m x n 
X = np.array([
    [150, 70], # it's our boy Jimmy again! 150 pounds, 70 inches tall. 
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

print(X.shape) # prints (10, 2)

#We need to transpose it because we want it to be n x m for our fast feed process
A0 = X.T
print(A0.shape) # prints (2, 10)

#raw vector of labels (solutions) in 1D
y = np.array([
    0,  # whew, thank God Jimmy isn't at risk for cardiovascular disease.
    1,   # damn, this guy wasn't as lucky
    1, # ok, this guy should have seen it coming. 5"8, 312 lbs isn't great.
    0,
    0,
    1,
    1,
    0,
    1,
    0
])
m = 10

#reshape the labels to be a 2D 1 x m vector to match A^[L]
Y = y.reshape(n[3], m)
Y.shape

#create the activation function
#np.expo takes the matrix and then models the function e^x
def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))

#start the feed forward process
m = 10
# layer 1 calculations

Z1 = W1 @ A0 + b1  # the @ means matrix multiplication

assert Z1.shape == (n[1], m) # just checking if shapes are good
A1 = sigmoid(Z1)

# layer 2 calculations
Z2 = W2 @ A1 + b2
assert Z2.shape == (n[2], m)
A2 = sigmoid(Z2)

# layer 3 calculations
Z3 = W3 @ A2 + b3
assert Z3.shape == (n[3], m)
A3 = sigmoid(Z3)

print(A3.shape)
y_hat = A3
print(y_hat) #right now all the predictions is around 50% because the weighting and the baises were random

