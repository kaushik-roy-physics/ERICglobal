import cupy as cp
import numpy as np
import json
import time

cp.random.seed(12345)

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
    return cp.asnumpy(theta_osc), cp.asnumpy(H_theta)  
   

# define the Kuramoto model parameters
N = 10000  # number of oscillators
K = 4.0  # coupling strength
#L_values = [0, 0.3, 0.5, 0.8]  # relative strengths
#L_values = [1.0, 1.5, 2 , 2.5]
#L_values = [3.0, 4.0, 5.0, 7.0]


# define the simulation parameters
T = 1000  # Integration time
dt = 0.1  # Timestep
tsteps = int(T / dt)  # total number of steps


# initialize the phase and natural frequency arrays
omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

# simulate the Kuramoto model and compute H_theta for different L values
theta_osc = []
H_theta = []

# simulate the Kuramoto model and compute H_theta for different L values
theta_osc = []
H_theta = []

start_time = time.time()

for j, L in enumerate(L_values):
    theta, H = calculate_order_function(theta_in, omega_in, K, L, N, T, dt)
    theta_osc.append(theta.tolist())  # Convert NumPy arrays to Python lists
    H_theta.append(H.tolist())        # Convert NumPy arrays to Python lists

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")

# Save the computed data to a JSON file
data_to_save = {
    'theta_osc': theta_osc,
    'H_theta': H_theta,
    'L_values': L_values
}

# Save data to a JSON file
output_filename = "daidoorderfunctionK4Lmedium.json"

with open(output_filename, 'w') as f:
    json.dump(data_to_save, f)
    
print("Data saved to", output_filename)

