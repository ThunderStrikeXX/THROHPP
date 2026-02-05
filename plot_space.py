import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from io import StringIO
import textwrap

def safe_loadtxt(filename, fill_value=-1e9):
    def parse_line(line):
        return (line.replace('-nan(ind)', str(fill_value))
                    .replace('nan', str(fill_value))
                    .replace('NaN', str(fill_value)))
    with open(filename, 'r') as f:
        lines = [parse_line(l) for l in f]
    return np.loadtxt(StringIO(''.join(lines)))

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
root = script_dir

cases = [d for d in os.listdir(root) if os.path.isdir(d) and "case" in d]

if len(cases) == 0:
    print("No case folders found")
    sys.exit(1)

print("Available cases:")
for i, c in enumerate(cases):
    print(i, c)

idx = int(input("Select case index: "))
case = cases[idx]

# -------------------- Files --------------------
x_file = os.path.join(case, "mesh.txt")
time_file = os.path.join(case, "time.txt")

targets = [
    "vapor_temperature.txt",
    "vapor_velocity.txt",
    "vapor_pressure.txt",
    "vapor_alpha.txt",
    "liquid_temperature.txt",
    "liquid_velocity.txt",
    "liquid_pressure.txt",
    "liquid_alpha.txt",
    "rho_vapor.txt",
    "liquid_rho.txt",
    "wall_temperature.txt",

    "gamma_xv.txt",
    "phi_xv.txt",
    "heat_source_wall_liquid_flux.txt",
    "heat_source_liquid_wall_flux.txt",
    "heat_source_vapor_liquid_phase.txt",
    "heat_source_liquid_vapor_phase.txt",
    "heat_source_vapor_liquid_flux.txt",
    "heat_source_liquid_vapor_flux.txt",
    "p_saturation.txt",
    "T_sur.txt",
    "delta_p_capillary.txt",
    "q_pp.txt"
]


y_files = [os.path.join(case, p) for p in targets]

for f in [x_file, time_file] + y_files:
    if not os.path.isfile(f):
        print("Missing file:", f)
        sys.exit(1)

# -------------------- Load data --------------------
x = safe_loadtxt(x_file)
time = safe_loadtxt(time_file)          # vettore tempo reale
Y = [safe_loadtxt(f) for f in y_files]

names = [
    "Vapor temperature",
    "Vapor velocity",
    "Vapor pressure",
    "Vapor volume fraction",
    "Liquid temperature",
    "Liquid velocity",
    "Liquid pressure",
    "Liquid volume fraction",
    "Vapor density",
    "Liquid density",
    "Wall temperature",

    "Gamma_xv",
    "Phi_xv",
    "Heat source wall→liquid (flux)",
    "Heat source liquid→wall (flux)",
    "Heat source vapor→liquid (phase)",
    "Heat source liquid→vapor (phase)",
    "Heat source vapor→liquid (flux)",
    "Heat source liquid→vapor (flux)",
    "Saturation pressure",
    "T_sur",
    "DPcap",
    "Q to wall"
]


units = [
    "[K]",
    "[m/s]",
    "[Pa]",
    "[-]",
    "[K]",
    "[m/s]",
    "[Pa]",
    "[-]",
    "[kg/m³]",
    "[kg/m³]",
    "[K]",

    "[kg/(m³·s)]",         # Gamma_xv
    "[kg/(m²·s)]",         # Phi_xv
    "[W/m³]",              # heat wall→liquid
    "[W/m³]",              # heat liquid→wall
    "[W/m³]",              # heat vapor→liquid (phase)
    "[W/m³]",              # heat liquid→vapor (phase)
    "[W/m³]",              # heat vapor→liquid (flux)
    "[W/m³]",              # heat liquid→vapor (flux)
    "[Pa]",                # psat
    "[K]",                  # T_sur
    "[Pa]",
    "[W/m3]"
]



# -------------------- Utils --------------------
def robust_ylim(y):
    vals = y.flatten() if y.ndim > 1 else y
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = np.min(vals), np.max(vals)
    margin = 0.1 * (hi - lo)
    return lo - margin, hi + margin

def time_to_index(t):
    return np.searchsorted(time, t, side='left')

def index_to_time(i):
    return time[i]

# -------------------- Figure --------------------
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.58)
line, = ax.plot([], [], lw=2, marker='o', markersize=4)
ax.grid(True)
ax.set_xlabel("Axial length [m]")

# Slider con valori temporali reali
ax_slider = plt.axes([0.15, 0.1, 0.55, 0.03])
slider = Slider(ax_slider, "Time [s]", time.min(), time.max(), valinit=time[0])

# -------------------- Buttons list --------------------
buttons = []
n_vars = len(Y)
n_cols = 3
button_width = 0.10
button_height = 0.08
col_gap = 0.02
row_gap = 0.10
start_x = 0.62
start_y = 0.86

for i, name in enumerate(names):
    col = i % n_cols
    row = i // n_cols
    label = "\n".join(textwrap.wrap(name, 15))
    x_pos = start_x + col * (button_width + col_gap)
    y_pos = start_y - row * row_gap
    b_ax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(b_ax, label, hovercolor='0.975')
    btn.label.set_fontsize(8)
    buttons.append(btn)

# Control buttons
ax_play = plt.axes([0.15, 0.02, 0.10, 0.05])
btn_play = Button(ax_play, "Play", hovercolor='0.975')
ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
btn_pause = Button(ax_pause, "Pause", hovercolor='0.975')
ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
btn_reset = Button(ax_reset, "Reset", hovercolor='0.975')

current_idx = 0
ydata = Y[current_idx]
n_frames = len(time)

ax.set_title(f"{names[current_idx]} {units[current_idx]}")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(*robust_ylim(ydata))

paused = [False]
current_frame = [0]

# -------------------- Drawing --------------------
def draw_frame(i, update_slider=True):
    y = Y[current_idx]
    if y.ndim > 1:
        line.set_data(x, y[i, :])
    else:
        line.set_data(x, y)

    if update_slider:
        slider.disconnect(slider_cid)
        slider.set_val(index_to_time(i))
        connect_slider()
    return line,

def update_auto(i):
    if not paused[0]:
        current_frame[0] = i
        draw_frame(i)
    return line,

def slider_update(val):
    i = time_to_index(val)
    current_frame[0] = i
    draw_frame(i, update_slider=False)
    fig.canvas.draw_idle()

def connect_slider():
    global slider_cid
    slider_cid = slider.on_changed(slider_update)

connect_slider()

# -------------------- Variable change --------------------
def change_variable(idx):
    global current_idx
    global ydata
    current_idx = idx
    ydata = Y[idx]
    ax.set_title(f"{names[idx]} {units[idx]}")
    ax.set_ylim(*robust_ylim(ydata))
    current_frame[0] = 0
    draw_frame(0)

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

# -------------------- Controls --------------------
def pause(event):
    paused[0] = True

def reset(event):
    paused[0] = True
    current_frame[0] = 0
    draw_frame(0)
    slider.set_val(time[0])
    fig.canvas.draw_idle()

def play(event):
    paused[0] = False

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)

# -------------------- Animation --------------------
skip = max(1, n_frames // 200)
ani = FuncAnimation(
    fig,
    update_auto,
    frames=range(0, n_frames, skip),
    interval=10000 / (n_frames/skip),
    blit=False,
    repeat=True
)

plt.show()
