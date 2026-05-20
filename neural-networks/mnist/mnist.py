import numpy as np

SEED = 42
np.random.seed(SEED)

def mse(w, b):
    return

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def grad_cost(activation, y):
    return activation - y  

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
        L = self.num_layers
        weighted_input = []
        activations = [x]

        output = x
        for i in range(0, self.num_layers - 1):
            z_temp = self.weights[i] @ output + self.biases[i]
            a_temp = sigmoid(z_temp)
            weighted_input.append(z_temp)
            activations.append(a_temp)
            output = a_temp

        # compute the error in last layer
        delta_L = grad_cost(activations[-1], y) * sigmoid_prime(weighted_input[-1])
        
        # backpropagation of the errors
        delta = [0 for _ in range(1, L + 1)] # index 0 is unused
        delta[-1] = delta_L
        for l in range(2, L):
            delta[-l] = self.weights[-l+1].T @ delta[-l+1] * sigmoid_prime(weighted_input[-l])

        return [delta[i] for i in range(1, L)], [delta[i] @ activations[i-1].T for i in range(1, L)]

    def SGD(self, lr, batch_size, epoch_nb, training_data, test_data):
        print('Starting SGD...\n')
        n_training = len(training_data)
        n_test = len(test_data)

        history = [(self.biases, self.weights)]
        # last batch can be smaller than the others
        for i in range(epoch_nb):
            np.random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n_training, batch_size)]
            for batch in batches:                
                # conpute the sum of derivative for each w and b 
                sum_grad_b = [np.zeros(b.shape) for b in self.biases]
                sum_grad_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in batch:
                    grad_b, grad_w = self.backprop(x, y)
                    sum_grad_b = [nb+dnb for nb, dnb in zip(sum_grad_b, grad_b)]
                    sum_grad_w = [nw+dnw for nw, dnw in zip(sum_grad_w, grad_w)]

                for b, j in zip(self.biases, range(len(self.biases))):
                    self.biases[j] = b - (lr / len(batch)) * sum_grad_b[j]
                
                for w, k in zip(self.weights, range(len(self.weights))):
                    self.weights[k] = w - (lr / len(batch)) * sum_grad_w[k]

                history.append((self.biases, self.weights))

            print("Epoch ", i, "/", epoch_nb-1, " done:", self.evaluate(test_data), "/", n_test)

        return history
    
    def evaluate(self, test_data):
        good_count = 0
        for x, y in test_data:
            if np.argmax(self.feedforward(x)) == y:
                good_count += 1
        
        return good_count
    
n = Network([784, 30, 10])

import data_loader
training_data, validation_data, test_data = data_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
n.SGD(3, 10, 30, training_data, test_data)