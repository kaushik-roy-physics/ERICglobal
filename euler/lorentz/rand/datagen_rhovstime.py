import cupy as cp
import time
import json


# Computing dtheta/dt or the r.h.s of the phase equation using CuPy

def dtheta_dt(theta, omega, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta**2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1/N) * K * sin_theta_sum) + ((1/N) * K * L * sinsq_theta_sum)
    return dtheta_dt


# Function to compute the rho_1 and rho_2 as a function of time

def generate_rhos(theta, omega, K, L, N, dt, tsteps):
    """Function to return rho_1, rho_2"""
    rho_1 = cp.zeros(tsteps)
    rho_2 = cp.zeros(tsteps)
    
    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    
    for i in range(tsteps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
        
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)
        
        r1 = cp.abs(cp.sum(theta_exp)/ N)
        r2 = cp.abs(cp.sum(theta_exp_2) / N)
        rho_1[i] = r1
        rho_2[i] = r2
        
    return cp.asnumpy(rho_1), cp.asnumpy(rho_2)  # Convert back to NumPy arrays for plotting


# Function to save the data to a JSON file
def save_data_to_json(rhos, L_values, filename):
    data = {
        "L_values": L_values,
        "rhos": [ {"rho_1": rho_1.tolist(), "rho_2": rho_2.tolist()} for rho_1, rho_2 in rhos ]
    }
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # parameters of the model
    N = 10000
    K = 3
    L_values = [0.0, 1.0, 1.5, 2.0]

    # simulation parameters
    T = 1000
    dt = 0.1
    tsteps = int(T / dt)

    # initial phase and frequency arrays
    cp.random.seed(12345)  # Use CuPy random generator
    omega_in = cp.random.standard_cauchy(N)
    theta_in = cp.random.uniform(-cp.pi, cp.pi, N)
    
    # Define the output filename
    output_filename = "rhovstimeK3Lmedium.json"
    
    start_time = time.time()

    # generate rhos for each L value
    rhos = []
    for L in L_values:
        rho_1, rho_2 = generate_rhos(theta_in, omega_in, K, L, N, dt, tsteps)
        rhos.append((rho_1, rho_2))
    
    save_data_to_json(rhos, L_values, output_filename)
    
    end_time = time.time()

    print("GPU computation took", end_time - start_time, "seconds")

    

    

