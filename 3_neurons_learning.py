from math import exp
from random import uniform

X = [i/1000 for i in range(1001)]
Y = [0] * 500 + [1] * 501

def sigmoid(z):
    return 1/(1+exp(-z))

def forward(x, w1, b1, w2, b2):
    z1 = x * w1 + b1
    a2 = sigmoid(z1) 
    z2 = a2 * w2 + b2
    y = sigmoid(z2)
    return z1, a2, z2, y

def backward(x, y_true, a2, w2, y_pred):
    delta2 = 2*(y_pred - y_true)*y_pred*(1-y_pred)
    dw2 = delta2*a2
    db2 = delta2

    delta_1 = delta2*w2*a2*(1-a2)
    dw1 = delta_1*x
    db1 = delta_1
    return dw2, db2, dw1, db1

def train_network(X, Y, epochs, lr):
    w1, w2, b1, b2 = 0, 0, 0, 0
    history = [(w1, w2, b1, b2)]

    for i in range(epochs):
        # calcul the mean of the gradient of the loss over training (non stocastic gradient)
        # if we implement stochastic gradient, the parameters would be changed for each training data (or little batches)
        sum_dw2, sum_db2, sum_dw1, sum_db1 = 0, 0, 0, 0

        for j in range(len(X)):
            z1, a2, z2, y = forward(X[j], w1, b1, w2, b2)
            dw2, db2, dw1, db1 = backward(X[j], Y[j], a2, w2, y)
            sum_dw2 += dw2
            sum_db2 += db2
            sum_dw1 += dw1
            sum_db1 += db1
        
        sum_dw2, sum_db2, sum_dw1, sum_db1 = (1/len(X)) * sum_dw2, (1/len(X)) * sum_db2, (1/len(X)) * sum_dw1, (1/len(X)) * sum_db1

        # update parameters
        w2 = w2 - lr * sum_dw2
        b2 = b2 - lr * sum_db2
        w1 = w1 - lr * sum_dw1
        b1 = b1 - lr * sum_db1
        history.append((w1, w2, b1, b2))
    
    return history

history = train_network(X, Y, epochs=10000, lr=0.1)

w1, w2, b1, b2 = history[-1]
print(f"Paramètres finaux : w1={w1:.2f}, w2={w2:.2f}, b1={b1:.2f}, b2={b2:.2f}")
print(f"""Test : 0.0 {forward(0, w1, b1, w2, b2)[-1]} \n
      0.2 {forward(0.2, w1, b1, w2, b2)[-1]} \n
      0.4 {forward(0.4, w1, b1, w2, b2)[-1]} \n
      0.6 {forward(0.6, w1, b1, w2, b2)[-1]} \n
      0.8 {forward(0.8, w1, b1, w2, b2)[-1]} \n
      1.0 {forward(1, w1, b1, w2, b2)[-1]} \n""")

# Visualisation de l'évolution des paramètres
import matplotlib.pyplot as plt

# Extraire les paramètres
epochs = list(range(len(history)))
w1_hist = [h[0] for h in history]
w2_hist = [h[1] for h in history]
b1_hist = [h[2] for h in history]
b2_hist = [h[3] for h in history]

# Style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#2C3E50')
fig.suptitle('Évolution des paramètres pendant l\'entraînement',
             fontsize=14, fontweight='bold', color='white')

params = [
    (w1_hist, 'w1 (poids couche 1)', '#E74C3C'),
    (w2_hist, 'w2 (poids couche 2)', '#3498DB'),
    (b1_hist, 'b1 (biais couche 1)', '#F39C12'),
    (b2_hist, 'b2 (biais couche 2)', '#1ABC9C')
]

for ax, (data, title, color) in zip(axes.flat, params):
    ax.set_facecolor('#34495E')
    ax.plot(epochs, data, color=color, linewidth=2)
    ax.set_title(title, fontsize=11, color='white', fontweight='bold')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Valeur', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    # Afficher valeur finale
    ax.axhline(y=data[-1], color=color, linestyle='--', alpha=0.5)
    ax.text(len(epochs)*0.7, data[-1], f'Final: {data[-1]:.2f}',
            color='white', fontsize=9, bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()