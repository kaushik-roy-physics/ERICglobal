import cupy as cp
import numpy as np
import json
import time

cp.random.seed(12345)

def save_output_to_json(L_values, psi_diff, filename):
    output_data = {"L_values": cp.asnumpy(L_values).tolist(),
                   "psi_diff": cp.asnumpy(psi_diff).tolist()}
    
    with open(filename, "w") as f:
        json.dump(output_data, f)

def dtheta_dt(theta, omega, K,L,N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta**2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega +  ((1/N) * K * sin_theta_sum) + ((1/N) * K * L * sinsq_theta_sum )
    return dtheta_dt


def generate_psi_diff(theta, omega, K, L, N, dt, tsteps):
    """Function to return psi_diff"""
    z1 = 0.0
    z2 = 0.0
    
    psi_1 = 0.0
    psi_2 = 0.0
    psi_diff = 0.0
    
    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    
    for i in range(tsteps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
        
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)
        
        z1 = cp.sum(theta_exp)/ N
        z2 = cp.sum(theta_exp_2) / N
        psi_1 = cp.angle(z1)
        psi_2 = cp.angle(z2)
        psi_diff = cp.mod((psi_2 - 2 * psi_1) + cp.pi, 2 * cp.pi) - cp.pi

    return psi_diff

# define the Kuramoto model parameters
N = 10000

# define the simulation parameters
T = 1000  # Integration time
dt = 0.1  # Timestep
tsteps = int(T / dt)  # total number of steps

    
# Set the coupling strength
K = 2.0  # fixed K value
L_values = cp.linspace(0.0, 3.0, 100)  # range of L values

# Generate random samples from a standard Cauchy distribution for omega and theta
omega_in = cp.random.standard_normal(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

# Initialize order parameter arrays

psi_diff = cp.zeros(len(L_values))

start_time = time.time()

# Compute the order parameters for each value of L
for i, L in enumerate(L_values):
    psi_diff[i] = generate_psi_diff(theta_in, omega_in, K, L, N, dt, tsteps)

# Specify the desired filename for the JSON output
output_filename = "psidiffvsLK2.json"

# Save the output to JSON
save_output_to_json(L_values, psi_diff, output_filename)

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")
