# Neural Networks from scratch

Learning neural networks by building everything from scratch to understand how they work under the hood.  

## Learning Progress

### Two Neurons Network (`2_neurons_learning.py`)

A minimal neural network with just 2 neurons (1 input, 1 output) learning a simple classification task:
- **Task**: Return 0 if input < 0.5, return 1 otherwise
- **Architecture**: Single neuron with sigmoid activation
- **Training**: Gradient Descent or Stochastic Gradient Descent
- **Loss function**: Mean Squared Error (MSE)  

### Three Neurons Network (`3_neurons_learning.py`)

Expanding the 2 neurons architecture with a hidden layer. Only implemented with gradient descent.

### XOR 

A 5 neurons network learning the XOR table. A on-line learning have been used (stochastic gradient descent of batch size 1).

### MNIST 

The famous 3 layers neural network for handwrite digits recognition, made from scratch.