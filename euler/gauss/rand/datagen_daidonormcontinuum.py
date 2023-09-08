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


def calculate_order_parameters(K, dt, tsteps, omega, theta, L, N):
    transient_steps = int(0.9 * tsteps)
    non_transient_steps = tsteps - transient_steps

    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    rho_1 = 0.0
    rho_2 = 0.0
    
    H_norm_th = 0.0 

    for t in range(transient_steps): 
        theta += dtheta_dt(theta,omega,K,L,N) * dt
    for t in range(non_transient_steps): 
        theta += dtheta_dt(theta,omega,K,L,N) * dt
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)

        rho_1 += cp.abs(cp.sum(theta_exp) / N)  
        rho_2 += cp.abs(cp.sum(theta_exp_2) / N)
    rho_1 /= non_transient_steps
    rho_2 /= non_transient_steps
    
    H_norm_th = cp.sqrt( (1/2) * ( rho_1 ** 2 + ((L ** 2) * (rho_2 ** 2) / 4 ) ) )

    return H_norm_th

N = 10000
T = 1000 
dt = 0.1  
tsteps = int(T / dt)

K = cp.linspace(0, 5, 100)
K_cpu = K.get() # To use for plotting
L_values = [0.0, 0.5]

omega_in = cp.random.standard_normal(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

H_norm_values = []

start_time = time.time()

for L in L_values:
    H_norm = cp.zeros(len(K))

    for i in cp.arange(len(K)):
        H_norm [i] = calculate_order_parameters(K[i], dt, tsteps, omega_in, theta_in, L, N )

    H_norm_values.append(H_norm.get().tolist())  # Convert to Python list
    
end_time = time.time()
    
print("GPU computation took", end_time - start_time, "seconds")

# Save the computed data to a JSON file

data_to_save = {
    "K_cpu": K_cpu.tolist(),
    "H_norm_values": H_norm_values,
    "L_values": L_values
}

output_filename = "daidonorm_continuum_Lsmall.json"
with open(output_filename, 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to", output_filename)
