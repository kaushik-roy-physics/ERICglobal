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

def generate_psis(theta, omega, K, L, N, dt, tsteps):
    """Function to return psi_2 - 2 psi_1"""
    psi_1 = cp.zeros(tsteps)
    psi_2 = cp.zeros(tsteps)
    psi_diff = cp.zeros(tsteps)
    
    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    
    for i in range(tsteps):
        theta = cp.add(theta,dtheta_dt(theta, omega, K, L, N) * dt)
        
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)
        
        z1 = cp.sum(theta_exp)/ N
        z2 = cp.sum(theta_exp_2) / N
        psi_1[i] = cp.angle(z1)
        psi_2[i] = cp.angle(z2)
        psi_diff[i] = cp.mod( (psi_2[i] - 2* psi_1[i]) + cp.pi, 2 * cp.pi) - cp.pi
        
    return cp.asnumpy(psi_diff) # Convert back to NumPy arrays for plotting

# Function to save the data to a JSON file
def save_data_to_json(psis, L_values, filename):
    data = {
        "L_values": L_values,
        "psis": [{"psi_diff": psi_diff.tolist()} for psi_diff in psis]
    }
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # parameters of the model
    N = 10000
    K = 4.0
    L_values = [0.0, 3.0]

    # simulation parameters
    T = 1000
    dt = 0.1
    tsteps = int(T / dt)

    # initial phase and frequency arrays
    cp.random.seed(12345)  # Use CuPy random generator
    omega_in = cp.random.standard_cauchy(N)
    theta_in = cp.random.uniform(-cp.pi, cp.pi, N)
    
    # Define the output filename
    output_filename = "psidiffvstimeK4Lhigh.json"

    start_time = time.time()

    # generate rhos for each L value
    psis = []
    for L in L_values:
        psi_diff = generate_psis(theta_in, omega_in, K, L, N, dt, tsteps)
        psis.append(psi_diff)
        
        
    save_data_to_json(psis, L_values, output_filename)
    
    end_time = time.time()

    print("GPU computation took", end_time - start_time, "seconds")

