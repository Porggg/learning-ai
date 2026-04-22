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
    
    def backprop(self, x, y):
        # not implemented for now
        return [0 for _ in range(len(self.biases))], [0 for _ in range(len(self.weights))]

    def SGD(self, lr, batch_size, epoch_nb, training_data):
        n_training = len(training_data)

        history = [(self.biases, self.weights)]
        # last batch can be smaller than the others
        for i in range(epoch_nb):
            np.random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n_training, batch_size)]
            for batch in batches:
                # update biases and weigth using formula (20) and (21)
                
                # conpute the sum of derivative for each w and b 
                for x, y in batch:
                    grad_b, grad_w = self.backprop(x, y)

                for b, i in zip(self.biases, range(len(self.biases))):
                    self.biases[i] = b - (lr / len(batch)) * grad_b[i]
                
                for w, i in zip(self.weights, range(len(self.weights))):
                    self.weights[i] = w - (lr / len(batch)) * grad_w[i]

                history.append((self.biases, self.weights))

            print("Epoch ", i, "/", epoch_nb, " done.")

        return history
    
n = Network([2, 3, 4])

print(f"Number of layers: {n.num_layers}")
print(f"Layer sizes: {n.sizes}")
print("\nBiases:")
for i, b in enumerate(n.biases):
    print(f"  Layer {i+1}:\n{b}")
print("\nWeights:")
for i, w in enumerate(n.weights):
    print(f"  Layer {i+1}:\n{w}")
print("")

result = n.feedforward(np.array([[0.5],[-1.2]]))

n.SGD(0.3, 20, 20, [])