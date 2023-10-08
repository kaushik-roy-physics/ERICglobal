import cupy as cp
import numpy as np
import json
import time

cp.random.seed(12345)

def save_output_to_json(L_values, phi_vals, filename):
    output_data = {"L_values": cp.asnumpy(L_values).tolist(),
                   "rho_ratios": cp.asnumpy(rho_ratios).tolist()}
    
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


def generate_rhos(theta, omega, K, L, N, dt, tsteps):
    """Function to return rho_ratios"""
    
    rho_1 = 0.0
    rho_2 = 0.0
    rho_ratios = 0.0
    
    z1 = 0.0
    z2 = 0.0
    
    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    
    for i in range(tsteps):
        theta = cp.add(theta,dtheta_dt(theta, omega, K, L, N) * dt)
        
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)
        
        z1 = cp.sum(theta_exp)/ N
        z2 = cp.sum(theta_exp_2) / N
    rho_1 = cp.abs(z1)
    rho_2 = cp.abs(z2)
    
    rho_ratios = rho_2 / (rho_1 **2)
        
    return rho_ratios

# define the Kuramoto model parameters
N = 10000

# define the simulation parameters
T = 1000  # Integration time
dt = 0.1  # Timestep
tsteps = int(T / dt)  # total number of steps

    
# Set the coupling strength
K = 4.0  # fixed K value
L_values = cp.linspace(0, 3.0, 100)  # range of L values

# Generate random samples from a standard Cauchy distribution for omega and theta
omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

# Initialize order parameter arrays
rho_1 = cp.zeros(len(L_values))
rho_2 = cp.zeros(len(L_values))
rho_ratios = cp.zeros(len(L_values))

start_time = time.time()

# Compute the order parameters for each value of L
for i, L in enumerate(L_values):
    rho_ratios[i] = generate_rhos(theta_in, omega_in, K, L, N, dt, tsteps)
    
# Specify the desired filename for the JSON output
output_filename = "OAtestK4.json"

# Save the output to JSON
save_output_to_json(L_values, rho_ratios, output_filename)

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")

print("Data saved as", output_filename)
