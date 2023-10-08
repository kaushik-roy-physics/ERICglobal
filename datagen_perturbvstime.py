import cupy as cp
import json
import time

# Computing dtheta/dt or the r.h.s of the phase equation using CuPy

def dtheta_dt(theta, omega, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta**2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1/N) * K * sin_theta_sum) + ((1/N) * K * L * sinsq_theta_sum)
    return dtheta_dt


# Function to compute the perturbed rhos


def evolve_and_perturb(theta, omega, K, L, N, T0, dt, tsteps, alpha):
    """Evolve the phases and then perturb by epsilon."""
    # First, evolve to t=T0
    for i in range(int((T0/dt) + 1)):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
    
    # Store rho_1 and rho_2 at t=T0
    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    rho_1_T0 = cp.abs(cp.sum(theta_exp) / N)
    rho_2_T0 = cp.abs(cp.sum(theta_exp_2) / N)
    
    # Perturb theta
    epsilon = cp.random.uniform(0, alpha, N)
    theta += epsilon
    
    # Evolve for the next
    rho_1_delta = cp.zeros(tsteps)
    rho_2_delta = cp.zeros(tsteps)
    
    for i in range(tsteps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
        
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)
        
        r1 = cp.abs(cp.sum(theta_exp) / N) - rho_1_T0
        r2 = cp.abs(cp.sum(theta_exp_2) / N) - rho_2_T0
        rho_1_delta[i] = r1
        rho_2_delta[i] = r2
    
    return cp.asnumpy(rho_1_delta), cp.asnumpy(rho_2_delta)

if __name__ == "__main__":

   # parameters of the model
    N = 10000
    K = 1.0
    L = 0.5
    
    # simulation parameters
    T0 = 1000
    Tf = 2000
    dt = 0.1
    tsteps = int((Tf - T0) / dt) + 1
    
    # initial phase and frequency arrays
    cp.random.seed(12345)  # Use CuPy random generator
    omega_in = cp.random.standard_cauchy(N)
    theta_in = cp.random.uniform(-cp.pi, cp.pi, N)
    
    alpha_values = [0.001, 0.01, 0.1, 1]
    
    delta_rho_1_list = []
    delta_rho_2_list = []
    
    for alpha in alpha_values:
        start_time = time.time()
    
        # Evolve and perturb
        delta_rho_1, delta_rho_2 = evolve_and_perturb(theta_in, omega_in, K, L, N, T0, dt, tsteps, alpha)
        
        end_time = time.time()
        
        print(f"GPU computation for alpha={alpha} took", end_time - start_time, "seconds")
        
        delta_rho_1_list.append(delta_rho_1)
        delta_rho_2_list.append(delta_rho_2)

    # Convert lists and other data to Python-native lists (from numpy arrays)
    data = {
        'delta_rho_1_list': [list(item) for item in delta_rho_1_list],
        'delta_rho_2_list': [list(item) for item in delta_rho_2_list],
        'T0': T0,
        'Tf': Tf,
        'tsteps': tsteps,
        'alpha_values': alpha_values
    }

    output_filename = "deltarhovstimeK1Lhalf.json"
    
    with open(output_filename, 'w') as f:
        json.dump(data, f)

    print("Saved output as:", output_filename)
