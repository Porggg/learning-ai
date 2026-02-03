import random
from math import exp

def sigmoid(z):
    return 1/(1+exp(-z))

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

