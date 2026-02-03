import random
from math import exp

training_set = [i/1000 for i in range(1001)]
label = [0] * 500 + [1] * 501

def sigmoid(z):
    return 1/(1+exp(-z))

def predict(x, w, b):
    """x : valeur du neuronne d'entree
       w : poid du lien vers le neuronne de sortie
       b : biais du neuronne de sortie"""
    
    return sigmoid(x*w + b)

def loss_i(predicted, expected):
    return (predicted - expected) ** 2

def loss(w, b):
    """function to minimize
       here it is the Mean Squared Error
       maybe the Binary Cross Entropy would be better because we are classifying 
    """
    sum_of_error = 0
    for i in range(len(training_set)):
        sum_of_error += loss_i(predict(training_set[i], w, b), label[i])

    return (1/len(label)) * sum_of_error

def gradientOneExample(w, b, i):
    z_i = training_set[i] * w + b
    y_bar_i = sigmoid(z_i)

    return [2*(y_bar_i - label[i])*y_bar_i*(1-y_bar_i)*training_set[i], 
     2*(y_bar_i - label[i])*y_bar_i*(1-y_bar_i)]

def gradient(w, b): 
    """evaluate the gradient of f at point (w, b)"""
    sum_w = 0
    sum_b = 0
    for i in range(len(training_set)):
        z_i = training_set[i] * w + b
        y_bar_i = sigmoid(z_i)
        sum_w += (y_bar_i - label[i])*y_bar_i*(1-y_bar_i)*training_set[i]
        sum_b += (y_bar_i - label[i])*y_bar_i*(1-y_bar_i)

    return [(2/len(training_set)) * sum_w, (2/len(training_set)) * sum_b]

def gradientDescent(iterations, learning_rate):
    w = random.uniform(-2, 2)
    b = random.uniform(-2, 2)
    history = [(w, b)]

    for i in range(iterations):
        dw, db = gradient(w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        history.append((w, b))

    return history

def stochasticGradientDescent(iterations, learning_rate):
    w = random.uniform(-2, 2)
    b = random.uniform(-2, 2)
    history = [(w, b)]

    for i in range(iterations):
        indices = list(range(len(training_set)))
        random.shuffle(indices)
        for j in indices:
            dw, db = gradientOneExample(w, b, j)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            history.append((w, b))

    return history

history = gradientDescent(iterations=4000, learning_rate=50)

w_final, b_final = history[-1]
print(f"Résultat : w={w_final:.2f}, b={b_final:.2f}")
print(f"Loss finale : {loss(w_final, b_final):.6f}")

for x in [0.2, 0.5, 0.8]:
    print(f"predict({x}) = {predict(x, w_final, b_final):.4f}")

# Affichez la loss à différents moments
for i in [0, 100, 500, 700, 1000]:
    w, b = history[i]
    print(f"Iter {i}: w={w:.2f}, b={b:.2f}, Loss={loss(w, b):.6f}")