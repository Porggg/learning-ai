import numpy as np

def mse(w, b):
    return

def sigmoid(z):
    return 1/(1+np.exp(-z))

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, x):
        if self.sizes[0] != len(x):
            raise ValueError(f"Input size {len(x)} does not match network input layer size {self.sizes[0]}")

        output = x
        for i in range(0, self.num_layers - 1):
            # apply linear transformation and sigmoid activation
            # weights[i] has shape (next_layer_size, current_layer_size)
            output = sigmoid(self.weights[i] @ output + self.biases[i])

        return output
    
    def SGD(self, lr, batch_size, epoch_nb):
        for i in range(epoch_nb):
            pass
        return 

n = Network([2, 3, 4])

print(f"Number of layers: {n.num_layers}")
print(f"Layer sizes: {n.sizes}")
print("\nBiases:")
for i, b in enumerate(n.biases):
    print(f"  Layer {i+1}:\n{b}")
print("\nWeights:")
for i, w in enumerate(n.weights):
    print(f"  Layer {i+1}:\n{w}")

result = n.feedforward(np.array([[0.5],[-1.2]]))