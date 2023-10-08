import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

cp.random.seed(12345)

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (6, 4)

def dtheta_dt(theta, omega, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt

def H_daido(theta, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    cos_2theta = cp.cos(2 * (theta - theta[:, None]))

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    cos_2theta_sum = cp.sum(cos_2theta, axis=1)

    H_daido = - ((1 / N) * sin_theta_sum) + ((1 / N) * (L / 2) * cos_2theta_sum)
    return H_daido

def calculate_order_function(theta, omega, K, L ,N, T, dt):
    tsteps = int(T/dt)
   
    theta_osc = cp.zeros(N)
    H_theta = cp.zeros(N)
    for t in range(tsteps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
        theta_osc = cp.mod(cp.unwrap(theta) + cp.pi, 2 * cp.pi) - cp.pi
        H_theta = H_daido(theta, K, L, N)
    return theta_osc, H_theta    
   
def plot_order_function(theta_osc, H_theta, L_values):
    colors = plt.get_cmap("tab10").colors
    plt.figure()

    for j, L in enumerate(L_values):
        theta_np = cp.asnumpy(theta_osc[j])
        H_daido = cp.asnumpy(H_theta[j])
        plt.scatter(theta_np, H_daido, s=4, color=colors[j], label=f"$\Lambda$ = {L_values[j]}")

    plt.xlabel(r'$\theta_i$', fontsize = 18)
    plt.ylabel(r'$H(\theta_i)$', fontsize = 18)
    plt.legend()

    x_ticks = [-cp.pi, -cp.pi / 2, 0, cp.pi / 2, cp.pi]
    x_ticklabels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
    plt.xticks(x_ticks, x_ticklabels)

    plt.savefig('Daido_order_functions_K3_Lhigh.pdf')
    plt.show()

# Define function to update the frame in animation
def update(frame):
    ax.clear()
    T_current = T_start + frame * T_interval

    print(f"Simulating for T = {T_current}...")  # print the progress

    theta, H = calculate_order_function(theta_in, omega_in, K, L , N, T_current, dt)
    theta_np = cp.asnumpy(theta)
    H_daido = cp.asnumpy(H)
    ax.scatter(theta_np, H_daido, s=4, label=f"T = {T_current}")
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$\theta_i$', fontsize = 18)
    ax.set_ylabel(r'$H(\theta_i)$', fontsize = 18)
    x_ticks = [-cp.pi, -cp.pi / 2, 0, cp.pi / 2, cp.pi]
    x_ticklabels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    return ax,

# define the Kuramoto model parameters
N = 10000  # number of oscillators
K = 4.0  # coupling strength
L = 0.5  # relative strengths

# define the simulation parameters
T_start = 200
T_end = 500
T_interval = 10
dt = 0.1  # Timestep

# initialize the phase and natural frequency arrays
omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

fig, ax = plt.subplots(figsize=(6, 4))
ani = FuncAnimation(fig, update, frames=(T_end-T_start)//T_interval, repeat=False)

ani.save('H_daido_vs_theta_osc_K4Lhalf.mp4', writer='ffmpeg', fps=2)  # 2 frames per second
plt.show()
