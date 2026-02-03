import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def f(x):
    """function to minimize"""
    return x**2

def gradient(x):
    """gradient of f"""
    return 2*x

fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.35)

x = np.linspace(-10, 10, 400)
y = f(x)
ax.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Interactive gradient descent')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 100)

current_x = 8.0
history = [current_x]

point, = ax.plot(current_x, f(current_x), 'ro', markersize=12, label='Position actuelle')
path_line, = ax.plot([current_x], [f(current_x)], 'r--', alpha=0.5, linewidth=1)
path_points, = ax.plot([current_x], [f(current_x)], 'go', markersize=6, alpha=0.5)

annotation = ax.annotate(f'x={current_x:.3f}\nf(x)={f(current_x):.3f}',
                         xy=(current_x, f(current_x)),
                         xytext=(10, 20), textcoords='offset points',
                         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

ax_lr = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_start = plt.axes([0.2, 0.15, 0.6, 0.03])

slider_lr = Slider(ax_lr, 'Learning Rate', 0.01, 1.0, valinit=0.1, valstep=0.01)
slider_start = Slider(ax_start, 'Position Départ', -9.0, 9.0, valinit=8.0, valstep=0.1)

ax_step = plt.axes([0.2, 0.05, 0.15, 0.05])
ax_auto = plt.axes([0.4, 0.05, 0.15, 0.05])
ax_reset = plt.axes([0.6, 0.05, 0.15, 0.05])

btn_step = Button(ax_step, 'Step')
btn_auto = Button(ax_auto, 'Auto (10 steps)')
btn_reset = Button(ax_reset, 'Reset')

def update_display():
    global current_x, history

    point.set_data([current_x], [f(current_x)])

    hist_y = [f(xi) for xi in history]
    path_line.set_data(history, hist_y)
    path_points.set_data(history, hist_y)

    annotation.set_text(f'x={current_x:.4f}\nf(x)={f(current_x):.4f}\nItération: {len(history)-1}')
    annotation.xy = (current_x, f(current_x))

    fig.canvas.draw_idle()

def step(event):
    global current_x, history
    lr = slider_lr.val
    current_x = current_x - lr * gradient(current_x)
    history.append(current_x)
    update_display()

def auto_steps(event):
    for _ in range(10):
        step(None)

def reset(event):
    global current_x, history
    current_x = slider_start.val
    history = [current_x]
    update_display()

def update_start(val):
    reset(None)

btn_step.on_clicked(step)
btn_auto.on_clicked(auto_steps)
btn_reset.on_clicked(reset)
slider_start.on_changed(update_start)

plt.show()
