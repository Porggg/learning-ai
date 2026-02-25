import random
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def loss(w11, w12, w21, w22, w31, w32, b1, b2, b3):
    """function to minimize
       here it is the Mean Squared Error
    """
    sum_of_error = 0
    for i in range(len(X)):
        sum_of_error += loss_i(forward(X[i][0], X[i][1], w11, w12, w21, w22, w31, w32, b1, b2, b3)['y'], Y[i])

    return (1/len(Y)) * sum_of_error

def backward(x1, x2, y_true, h1, h2, y_pred, w31, w32):
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
    random.seed(42)  # Fix seed for same initialization across all trainings
    # init the weights and biases
    w11, w12, w21, w22, w31, w32, b1, b2, b3 = [random.uniform(-1, 1) for _ in range(9)]
    
    for i in range(nb_epoch):
        for j in range(len(X)):
            f = forward(X[j][0], X[j][1], w11, w12, w21, w22, w31, w32, b1, b2, b3)
            dw11, dw12, dw21, dw22, dw31, dw32, db1, db2, db3 = backward(X[j][0], X[j][1], Y[j], f['h1'], f['h2'], f['y'], w31, w32)

            # update the params 
            w11 = w11 - lr * dw11
            w12 = w12 - lr * dw12
            w21 = w21 - lr * dw21
            w22 = w22 - lr * dw22
            w31 = w31 - lr * dw31
            w32 = w32 - lr * dw32
            b1 = b1 - lr * db1
            b2 = b2 - lr * db2
            b3 = b3 - lr * db3

    final_loss = loss(w11, w12, w21, w22, w31, w32, b1, b2, b3)
    return final_loss

# Define ranges
nb_epochs_list = np.arange(100, 10001, 100)  # 100, 200, 300, ..., 10000 (100 values)
lr_list = np.arange(0.1, 1.01, 0.02)  # 0.1, 0.12, ..., 1.0 (46 values)

# Initialize loss matrix
losses = np.zeros((len(nb_epochs_list), len(lr_list)))

# Compute losses
for i, nb_epoch in enumerate(nb_epochs_list):
    for j, lr_val in enumerate(lr_list):
        losses[i, j] = train(nb_epoch, lr_val)
        print(f"Epochs: {nb_epoch}, LR: {lr_val:.2f}, Loss: {losses[i, j]:.6f}")

# Plot 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

X_mesh, Y_mesh = np.meshgrid(lr_list, nb_epochs_list)
surf = ax.plot_surface(X_mesh, Y_mesh, losses, cmap='viridis', edgecolor='none')

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Number of Epochs')
ax.set_zlabel('Final Loss')
ax.set_title('3D Surface of Final Loss vs Learning Rate and Number of Epochs')

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Final Loss')

plt.show()