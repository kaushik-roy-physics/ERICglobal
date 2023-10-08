import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cp.random.seed(12345)

def dtheta_dt(theta, omega, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = cp.sin(theta - theta[:, None]) ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt

N = 500
K = 4.0
L = 0.5

T = 1000
dt = 0.1
tsteps = int(T/dt)

omega_in = cp.random.standard_cauchy(N)
theta = cp.random.uniform(-cp.pi, cp.pi, N)

theta_osc = cp.zeros(N)

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)


# Define a colormap for coloring the phases
cmap = plt.get_cmap('hsv', N)

# Create a scatter plot with initial colors based on phases
x = 2.0 * cp.cos(theta)
y = 2.0 * cp.sin(theta)
sc = ax.scatter(cp.asnumpy(x), cp.asnumpy(y), s=10, c=cp.asnumpy(theta), cmap=cmap)

# Initialize time text for the animation
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

#def animate(i):
#    global theta
#    theta += dtheta_dt(theta, omega_in, K, L, N) * dt
#    theta_osc = (theta + cp.pi) % (2 * cp.pi) - cp.pi
#    x = 2.0 * cp.cos(theta_osc)
#    y = 2.0 * cp.sin(theta_osc)
#    sc.set_offsets(cp.asnumpy(cp.vstack((x, y)).T))
    
    # Update the time text in the animation
#    time_text.set_text(f'Time: {i * dt:.1f} s')
    
    # Print progress updates to the terminal
#    if i % 100 == 0:
#        print(f'Simulation Progress: {i / tsteps * 100:.1f}%')


def animate(i):
    global theta
    theta += dtheta_dt(theta, omega_in, K, L, N) * dt
    theta_osc = (theta + cp.pi) % (2 * cp.pi) - cp.pi
    x = 2.0 * cp.cos(theta_osc)
    y = 2.0 * cp.sin(theta_osc)
    
    # Update the scatter plot data and colors based on new phases
    sc.set_offsets(cp.asnumpy(cp.vstack((x, y)).T))
    sc.set_array(cp.asnumpy(theta_osc))
    
    # Update the time text in the animation
    time_text.set_text(f'Time: {i * dt:.1f} s')

    # Print progress bar to the terminal
    progress = (i + 1) / tsteps
    bar_length = 40
    block = int(round(bar_length * progress))
    progress_bar = "|" + "=" * block + "-" * (bar_length - block) + f"| {progress * 100:.1f}%"
    print(progress_bar, end='\r')
    if i == tsteps - 1:
        print()  # Print a newline after the progress bar is complete

    return sc, time_text

# Create a colorbar for the phases
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Phase')

# Create the animation
ani = FuncAnimation(fig, animate, frames=tsteps, blit=True)

# Save the animation as an .mp4 file
ani.save('phasesvstime_K4Lsmall.mp4', writer='ffmpeg', fps=10)

plt.show()
