import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np

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
    # init the weights and biases
    random.seed(42)
    w11, w12, w21, w22, w31, w32, b1, b2, b3 = [random.uniform(-1, 1) for _ in range(9)]
    
    weight_history = []
    bias_history = []
    output_history = []
    
    for i in range(nb_epoch):
        # Record current weights, biases, and outputs before updating
        weight_history.append([w11, w12, w21, w22, w31, w32])
        bias_history.append([b1, b2, b3])
        current_outputs = [forward(x[0], x[1], w11, w12, w21, w22, w31, w32, b1, b2, b3)['y'] for x in X]
        output_history.append(current_outputs)
        
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

    return w11, w12, w21, w22, w31, w32, b1, b2, b3, weight_history, bias_history, output_history

# Train the network
nb_epochs = 1500
lr = 0.55
w11, w12, w21, w22, w31, w32, b1, b2, b3, weight_hist, bias_hist, output_hist = train(nb_epochs, lr)
final_weights = (w11, w12, w21, w22, w31, w32, b1, b2, b3)

# Convert histories to numpy arrays for easier plotting
weight_hist = np.array(weight_hist)
bias_hist = np.array(bias_hist)
output_hist = np.array(output_hist)

epochs = np.arange(nb_epochs)

# Plot weights evolution
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(epochs, weight_hist[:, 0], label='w11')
plt.plot(epochs, weight_hist[:, 1], label='w12')
plt.plot(epochs, weight_hist[:, 2], label='w21')
plt.plot(epochs, weight_hist[:, 3], label='w22')
plt.plot(epochs, weight_hist[:, 4], label='w31')
plt.plot(epochs, weight_hist[:, 5], label='w32')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.title('Evolution of Weights During Training')
plt.legend()
plt.grid(True)

# Plot biases evolution
plt.subplot(3, 1, 2)
plt.plot(epochs, bias_hist[:, 0], label='b1 (hidden1 bias)')
plt.plot(epochs, bias_hist[:, 1], label='b2 (hidden2 bias)')
plt.plot(epochs, bias_hist[:, 2], label='b3 (output bias)')
plt.xlabel('Epochs')
plt.ylabel('Bias Value')
plt.title('Evolution of Biases During Training')
plt.legend()
plt.grid(True)

# Plot outputs evolution
plt.subplot(3, 1, 3)
for i, (x, expected) in enumerate(zip(X, Y)):
    plt.plot(epochs, output_hist[:, i], label=f'Output for input {x} (expected: {expected})')
plt.xlabel('Epochs')
plt.ylabel('Output Value')
plt.title('Evolution of Network Outputs During Training')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final predictions
print("Final predictions:")
for j, (x, expected) in enumerate(zip(X, Y)):
    pred = forward(x[0], x[1], *final_weights)['y']
    print(f"Input {x}: predicted {pred:.4f}, expected {expected}")
print(f"Final loss: {loss(*final_weights):.6f}")