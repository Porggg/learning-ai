import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


training_set = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(x, w, b): 
    """x : valeur du neuronne d'entree
       w : poid du lien vers le neuronne de sortie
       b : biais du neuronne de sortie"""
    
    return sigmoid(x*w + b)

def loss_i(predicted, expected):
    return (predicted - expected) ** 2

def f(x, y):
    """function to minimize"""
    return x**2 + y**2

#def f(w, b):
    """function to minimize
       here it is the Mean Squared Error
       maybe the Binary Cross Entropy would be better because we are classifying
    """
    sum_of_error = 0
    for i in range(len(training_set)):
        sum_of_error += loss_i(predict(training_set[i], w, b), label[i])

    return (1/len(label)) * sum_of_error

def f_vectorized(W, B):
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = f(W[i, j], B[i, j])
    return Z

def gradient(x, y):
    return [2*x, 2*y]

# def gradient(w, b): 
    """evaluate the gradient of f at point (w, b)"""
    sum_w = 0
    sum_b = 0
    for i in range(len(training_set)):
        z_i = training_set[i] * w + b
        y_bar_i = sigmoid(z_i)
        sum_w += (y_bar_i - label[i])*y_bar_i*(1-y_bar_i)*training_set[i]
        sum_b += (y_bar_i - label[i])*y_bar_i*(1-y_bar_i)

    return [(2/len(training_set)) * sum_w, (2/len(training_set)) * sum_b]

def gradientOneExample(w, b, i):
    z_i = training_set[i] * w + b
    y_bar_i = sigmoid(z_i)

    return [2*(y_bar_i - label[i])*y_bar_i*(1-y_bar_i)*training_set[i], 
     2*(y_bar_i - label[i])*y_bar_i*(1-y_bar_i)]

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'gd': '#E74C3C',       
    'sgd': '#3498DB',      
    'bg': '#2C3E50',       
    'accent': '#1ABC9C',   
    'text': '#ECF0F1',     
    'btn': '#34495E',      
    'btn_play': '#27AE60', 
    'btn_reset': '#E67E22' 
}

fig = plt.figure(figsize=(16, 9), facecolor=COLORS['bg'])
fig.suptitle('Gradient Descent vs Stochastic Gradient Descent',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)

ax3d = fig.add_subplot(121, projection='3d', facecolor=COLORS['bg'])

ax2d = fig.add_subplot(122, facecolor=COLORS['bg'])

plt.subplots_adjust(bottom=0.28, wspace=0.25, top=0.92, left=0.05, right=0.95)

x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)
W, B = np.meshgrid(x, y)
Z = f_vectorized(W, B)

ax3d.plot_surface(W, B, Z, cmap='plasma', alpha=0.8, edgecolor='none')
ax3d.set_xlabel('w', fontsize=10, color=COLORS['text'])
ax3d.set_ylabel('b', fontsize=10, color=COLORS['text'])
ax3d.set_zlabel('Loss', fontsize=10, color=COLORS['text'])
ax3d.tick_params(colors=COLORS['text'])

contour = ax2d.contour(W, B, Z, levels=30, cmap='plasma', alpha=0.8)
ax2d.contourf(W, B, Z, levels=30, cmap='plasma', alpha=0.4)
ax2d.set_xlabel('w', fontsize=10, color=COLORS['text'])
ax2d.set_ylabel('b', fontsize=10, color=COLORS['text'])
ax2d.set_title('Vue du dessus (Contour)', fontsize=12, color=COLORS['text'], pad=10)
ax2d.set_aspect('equal')
ax2d.tick_params(colors=COLORS['text'])
cbar = plt.colorbar(contour, ax=ax2d, shrink=0.8)
cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
cbar.outline.set_edgecolor(COLORS['text'])
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text'])

start_x, start_y = 2.0, 2.0

gd_x, gd_y = start_x, start_y
sgd_x, sgd_y = start_x, start_y
history_gd = [(gd_x, gd_y)]
history_sgd = [(sgd_x, sgd_y)]
stochastic_index = 0

point_gd_3d, = ax3d.plot([gd_x], [gd_y], [f(gd_x, gd_y)],
                          'o', color=COLORS['gd'], markersize=12, zorder=10,
                          markeredgecolor='white', markeredgewidth=2)
path_gd_3d, = ax3d.plot([gd_x], [gd_y], [f(gd_x, gd_y)],
                         '-', color=COLORS['gd'], linewidth=2.5, alpha=0.9)

point_gd_2d, = ax2d.plot(gd_x, gd_y, 'o', color=COLORS['gd'], markersize=14,
                          markeredgecolor='white', markeredgewidth=2)
path_gd_2d, = ax2d.plot([gd_x], [gd_y], '-', color=COLORS['gd'], linewidth=2.5,
                         alpha=0.9, label='GD')

point_sgd_3d, = ax3d.plot([sgd_x], [sgd_y], [f(sgd_x, sgd_y)],
                           'o', color=COLORS['sgd'], markersize=12, zorder=10,
                           markeredgecolor='white', markeredgewidth=2)
path_sgd_3d, = ax3d.plot([sgd_x], [sgd_y], [f(sgd_x, sgd_y)],
                          '-', color=COLORS['sgd'], linewidth=2.5, alpha=0.9)

point_sgd_2d, = ax2d.plot(sgd_x, sgd_y, 'o', color=COLORS['sgd'], markersize=12,
                           markeredgecolor='white', markeredgewidth=2)
path_sgd_2d, = ax2d.plot([sgd_x], [sgd_y], '-', color=COLORS['sgd'], linewidth=2.5,
                          alpha=0.9, label='Stochastic GD')

legend = ax2d.legend(loc='upper right', facecolor=COLORS['btn'], edgecolor=COLORS['text'],
                      fontsize=10, framealpha=0.9)
for text in legend.get_texts():
    text.set_color(COLORS['text'])

annotation_gd = ax2d.annotate(
    f'GD\nw={gd_x:.3f}  b={gd_y:.3f}\nLoss={f(gd_x, gd_y):.4f}\nIter: 0',
    xy=(gd_x, gd_y),
    xytext=(20, 20), textcoords='offset points',
    fontsize=9, color='white', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['gd'], alpha=0.95, edgecolor='white'))

annotation_sgd = ax2d.annotate(
    f'SGD\nw={sgd_x:.3f}  b={sgd_y:.3f}\nLoss={f(sgd_x, sgd_y):.4f}\nIter: 0',
    xy=(sgd_x, sgd_y),
    xytext=(20, -50), textcoords='offset points',
    fontsize=9, color='white', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['sgd'], alpha=0.95, edgecolor='white'))

slider_height = 0.025
slider_width = 0.5
slider_left = 0.25

ax_lr = plt.axes([slider_left, 0.19, slider_width, slider_height], facecolor=COLORS['btn'])
ax_start_x = plt.axes([slider_left, 0.14, slider_width, slider_height], facecolor=COLORS['btn'])
ax_start_y = plt.axes([slider_left, 0.09, slider_width, slider_height], facecolor=COLORS['btn'])

slider_lr = Slider(ax_lr, 'Learning Rate', 0.01, 100.0, valinit=0.1, valstep=0.1,
                   color=COLORS['accent'], initcolor='none')
slider_start_x = Slider(ax_start_x, 'Start W', -90.0, 90.0, valinit=2.0, valstep=1.0,
                        color=COLORS['accent'], initcolor='none')
slider_start_y = Slider(ax_start_y, 'Start B', -90.0, 90.0, valinit=2.0, valstep=1.0,
                        color=COLORS['accent'], initcolor='none')

for slider in [slider_lr, slider_start_x, slider_start_y]:
    slider.label.set_color(COLORS['text'])
    slider.label.set_fontsize(10)
    slider.label.set_fontweight('bold')
    slider.valtext.set_color(COLORS['text'])
    slider.valtext.set_fontsize(10)

btn_height = 0.04
btn_y = 0.025
btn_width = 0.08
btn_gap = 0.085
toggle_width = 0.06

total_width = 6 * btn_width + 5 * (btn_gap - btn_width) + 2 * toggle_width + 0.02
btn_start = (1 - total_width) / 2

ax_step = plt.axes([btn_start, btn_y, btn_width, btn_height])
ax_auto = plt.axes([btn_start + btn_gap, btn_y, btn_width, btn_height])
ax_auto50 = plt.axes([btn_start + 2*btn_gap, btn_y, btn_width, btn_height])
ax_auto500 = plt.axes([btn_start + 3*btn_gap, btn_y, btn_width, btn_height])
ax_play = plt.axes([btn_start + 4*btn_gap, btn_y, btn_width, btn_height])
ax_reset = plt.axes([btn_start + 5*btn_gap, btn_y, btn_width, btn_height])

ax_toggle_gd = plt.axes([btn_start + 6*btn_gap, btn_y, toggle_width, btn_height])
ax_toggle_sgd = plt.axes([btn_start + 6*btn_gap + btn_gap - btn_width + toggle_width, btn_y, toggle_width, btn_height])

btn_step = Button(ax_step, 'Step', color=COLORS['btn'], hovercolor=COLORS['accent'])
btn_auto = Button(ax_auto, '+10', color=COLORS['btn'], hovercolor=COLORS['accent'])
btn_auto50 = Button(ax_auto50, '+50', color=COLORS['btn'], hovercolor=COLORS['accent'])
btn_auto500 = Button(ax_auto500, '+500', color=COLORS['btn'], hovercolor=COLORS['accent'])
btn_play = Button(ax_play, 'Play', color=COLORS['btn_play'], hovercolor='#2ECC71')
btn_reset = Button(ax_reset, 'Reset', color=COLORS['btn_reset'], hovercolor='#F39C12')

btn_toggle_gd = Button(ax_toggle_gd, 'GD', color=COLORS['gd'], hovercolor='#C0392B')
btn_toggle_sgd = Button(ax_toggle_sgd, 'SGD', color=COLORS['sgd'], hovercolor='#2980B9')

for btn in [btn_step, btn_auto, btn_auto50, btn_auto500, btn_play, btn_reset]:
    btn.label.set_color('white')
    btn.label.set_fontweight('bold')
    btn.label.set_fontsize(10)

for btn in [btn_toggle_gd, btn_toggle_sgd]:
    btn.label.set_color('white')
    btn.label.set_fontweight('bold')
    btn.label.set_fontsize(10)

is_playing = False
anim = None
show_gd = True
show_sgd = False

# SGD hidden by default - set button to inactive state
btn_toggle_sgd.color = COLORS['btn']
btn_toggle_sgd.hovercolor = COLORS['accent']

# Hide SGD elements initially
point_sgd_3d.set_visible(False)
path_sgd_3d.set_visible(False)
point_sgd_2d.set_visible(False)
path_sgd_2d.set_visible(False)
annotation_sgd.set_visible(False)

def update_display():
    global gd_x, gd_y, sgd_x, sgd_y, history_gd, history_sgd

    gd_hist_x = [p[0] for p in history_gd]
    gd_hist_y = [p[1] for p in history_gd]
    gd_hist_z = [f(p[0], p[1]) for p in history_gd]

    sgd_hist_x = [p[0] for p in history_sgd]
    sgd_hist_y = [p[1] for p in history_sgd]
    sgd_hist_z = [f(p[0], p[1]) for p in history_sgd]

    point_gd_3d.set_data_3d([gd_x], [gd_y], [f(gd_x, gd_y)])
    path_gd_3d.set_data_3d(gd_hist_x, gd_hist_y, gd_hist_z)
    point_gd_2d.set_data([gd_x], [gd_y])
    path_gd_2d.set_data(gd_hist_x, gd_hist_y)

    point_sgd_3d.set_data_3d([sgd_x], [sgd_y], [f(sgd_x, sgd_y)])
    path_sgd_3d.set_data_3d(sgd_hist_x, sgd_hist_y, sgd_hist_z)
    point_sgd_2d.set_data([sgd_x], [sgd_y])
    path_sgd_2d.set_data(sgd_hist_x, sgd_hist_y)

    annotation_gd.set_text(
        f'GD\nw={gd_x:.4f}  b={gd_y:.4f}\nLoss={f(gd_x, gd_y):.6f}\nIter: {len(history_gd)-1}')
    annotation_gd.xy = (gd_x, gd_y)

    annotation_sgd.set_text(
        f'SGD\nw={sgd_x:.4f}  b={sgd_y:.4f}\nLoss={f(sgd_x, sgd_y):.6f}\nIter: {len(history_sgd)-1}')
    annotation_sgd.xy = (sgd_x, sgd_y)

    point_gd_3d.set_visible(show_gd)
    path_gd_3d.set_visible(show_gd)
    point_gd_2d.set_visible(show_gd)
    path_gd_2d.set_visible(show_gd)
    annotation_gd.set_visible(show_gd)

    point_sgd_3d.set_visible(show_sgd)
    path_sgd_3d.set_visible(show_sgd)
    point_sgd_2d.set_visible(show_sgd)
    path_sgd_2d.set_visible(show_sgd)
    annotation_sgd.set_visible(show_sgd)

    fig.canvas.draw_idle()

def step(event):
    global gd_x, gd_y, sgd_x, sgd_y, history_gd, history_sgd, stochastic_index
    lr = slider_lr.val

    grad_gd = gradient(gd_x, gd_y)
    gd_x = gd_x - lr * grad_gd[0]
    gd_y = gd_y - lr * grad_gd[1]
    history_gd.append((gd_x, gd_y))

    grad_sgd = gradientOneExample(sgd_x, sgd_y, stochastic_index)
    stochastic_index = (stochastic_index + 1) % len(training_set)
    sgd_x = sgd_x - lr * grad_sgd[0]
    sgd_y = sgd_y - lr * grad_sgd[1]
    history_sgd.append((sgd_x, sgd_y))

    update_display()

def toggle_gd(event):
    global show_gd
    show_gd = not show_gd
    if show_gd:
        btn_toggle_gd.color = COLORS['gd']
        btn_toggle_gd.hovercolor = '#C0392B'
    else:
        btn_toggle_gd.color = COLORS['btn']
        btn_toggle_gd.hovercolor = COLORS['accent']
    update_display()

def toggle_sgd(event):
    global show_sgd
    show_sgd = not show_sgd
    if show_sgd:
        btn_toggle_sgd.color = COLORS['sgd']
        btn_toggle_sgd.hovercolor = '#2980B9'
    else:
        btn_toggle_sgd.color = COLORS['btn']
        btn_toggle_sgd.hovercolor = COLORS['accent']
    update_display()

def auto_steps(event):
    for _ in range(10):
        step(None)

def auto_steps_50(event):
    for _ in range(50):
        step(None)

def auto_steps_500(event):
    for _ in range(500):
        step(None)

def animate(frame):
    global is_playing
    if is_playing:
        step(None)
    return point_gd_2d, path_gd_2d, point_sgd_2d, path_sgd_2d

def toggle_play(event):
    global is_playing, anim
    is_playing = not is_playing
    if is_playing:
        btn_play.label.set_text('Stop')
        btn_play.color = '#C0392B' 
        btn_play.hovercolor = '#E74C3C'
        anim = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
        plt.draw()
    else:
        btn_play.label.set_text('Play')
        btn_play.color = COLORS['btn_play']
        btn_play.hovercolor = '#2ECC71'
        if anim is not None:
            anim.event_source.stop()

def reset(event):
    global gd_x, gd_y, sgd_x, sgd_y, history_gd, history_sgd, stochastic_index
    gd_x = slider_start_x.val
    gd_y = slider_start_y.val
    sgd_x = slider_start_x.val
    sgd_y = slider_start_y.val
    history_gd = [(gd_x, gd_y)]
    history_sgd = [(sgd_x, sgd_y)]
    stochastic_index = 0
    update_display()

def update_start(val):
    reset(None)

btn_step.on_clicked(step)
btn_auto.on_clicked(auto_steps)
btn_auto50.on_clicked(auto_steps_50)
btn_auto500.on_clicked(auto_steps_500)
btn_play.on_clicked(toggle_play)
btn_reset.on_clicked(reset)
btn_toggle_gd.on_clicked(toggle_gd)
btn_toggle_sgd.on_clicked(toggle_sgd)
slider_start_x.on_changed(update_start)
slider_start_y.on_changed(update_start)

plt.show()
