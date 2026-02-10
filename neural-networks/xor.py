import random
from math import exp

X = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

Y = [0, 1, 1, 0]

def sigmoid(z):
    return 1/(1+exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward(x1, x2, w11, w12, w21, w22, w31, w32, b1, b2, b3):
    z1 = x1*w11 + x2*w12 + b1
    z2 = x1*w21 + x2*w22 + b2   
    h1 = sigmoid(z1)
    h2 = sigmoid(z2)
    
    z3 = h1*w31 + h2*w32 + b3
    y = sigmoid(z3)

    return {
    'z1': z1, 'h1': h1,
    'z2': z2, 'h2': h2,
    'z3': z3, 'y': y
    }  

def loss_i(predicted, expected):
    return (predicted - expected) ** 2

def loss(x1, x2, w11, w12, w21, w22, w31, w32, b1, b2, b3):
    """function to minimize
       here it is the Mean Squared Error
    """
    sum_of_error = 0
    for i in range(len(X)):
        sum_of_error += loss_i(forward(X[i][0], X[i][1], x1, x2, w11, w12, w21, w22, w31, w32, b1, b2, b3)['y'], Y[i])

    return (1/len(Y)) * sum_of_error

def backward(x1, x2, y_true, z1, h1, z2, h2, z3, y_pred, w31, w32):
    delta3 = 2*(y_pred - y_true)*y_pred*(1-y_pred)
    dw31 = delta3*h1
    dw32 = delta3*h2
    db3 = delta3

    delta2 = delta3 * w32 * h2 * (1 - h2)
    delta1 = delta3 * w31 * h1 * (1 - h1)
    dw21 = delta2 * x1
    dw22 = delta2 * x2
    dw11 = delta1 * x1
    dw12 = delta1 * x2
    db2 = delta2
    db1 = delta1

    return dw11, dw12, dw21, dw22, dw31, dw32, db1, db2, db3

def train(nb_epoch, lr):
    return 