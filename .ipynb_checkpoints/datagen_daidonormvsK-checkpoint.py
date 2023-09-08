import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import json

cp.random.seed(12345)

def dtheta_dt(theta, omega, K, L, N):
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt


def H_daido(theta, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    cos_2theta = cp.cos(2*(theta - theta[:, None]))

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    cos_2theta_sum = cp.sum(cos_2theta, axis=1)

    H_daido = - ( (1/N) * sin_theta_sum ) + ( (1/N) * (L/2) * cos_2theta_sum )
    return H_daido


def compute_daido_norm(K, L, N, T, dt, omega, theta):
    tsteps = int(T / dt)
    transient_steps = int(0.9 * tsteps)
    non_transient_steps = tsteps - transient_steps
    
    daido_norms = cp.array([])  # Initialize as a Cupy array

    for i in range(transient_steps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
    for i in range(non_transient_steps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt

#        if i >= transient_steps:
        H = cp.sqrt((1/N) * cp.sum(cp.abs(H_daido(theta, K, L, N))**2))
        daido_norms = cp.append(daido_norms, H)  # Append to Cupy array

    return cp.mean(daido_norms)

N = 10000
T = 1000
dt = 0.1
tsteps = int(T / dt)

K = cp.linspace(2.2, 2.5, 100)
K_cpu = K.get()  # To use for plotting
L_values = [0, 2.0, 3.0]

#L_values = [0, 0.1, 0.3, 0.5]
#L_values = [0, 1, 1.5, 2]
#L_values = [0, 3, 5, 7]

omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

H_norm_values = []

start_time = time.time()

for L in L_values:
    H_norm = cp.zeros(len(K))

    for i in cp.arange(len(K)):
        H_norm[i] = compute_daido_norm(K[i], L, N, T, dt, omega_in, theta_in)

    H_norm_values.append(H_norm.get().tolist())  # Convert to Python list

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")

# Save the computed data to a JSON file

data_to_save = {
    "K_cpu": K_cpu.tolist(),
    "H_norm_values": H_norm_values,
    "L_values": L_values
}

output_filename = "daidonormnearKLsmall.json"
with open(output_filename, 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to", output_filename)