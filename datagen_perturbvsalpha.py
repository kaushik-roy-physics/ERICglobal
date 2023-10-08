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


def evolve_and_perturb(theta, omega, K, L, N, Tp, dt, Tf, alpha):
    """Evolve the phases and then perturb by epsilon."""
    # First, evolve to t=Tp

    tsteps_1 = int(Tp/dt) + 1

    transient_steps_1 = int(0.9 * tsteps_1)
    non_transient_steps_1 = tsteps_1 - transient_steps_1

    theta_exp = cp.exp(1j * theta)
    theta_exp_2 = cp.exp(1j * 2 * theta)
    rho_1_Tp = 0.0
    rho_2_Tp = 0.0

    for i in range(transient_steps_1):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
    for i in range(non_transient_steps_1):
        theta += dtheta_dt(theta, omega, K, L, N) * dt

        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)

        rho_1_Tp += cp.abs(cp.sum(theta_exp) / N)  
        rho_2_Tp += cp.abs(cp.sum(theta_exp_2) / N)
    rho_1_Tp /= non_transient_steps_1
    rho_2_Tp /= non_transient_steps_1

    # Perturb theta
    epsilon = cp.random.uniform(0, alpha, N)
    theta += epsilon
    
    # Evolve for the next
    rho_1_Tf = 0.0
    rho_2_Tf = 0.0
    rho_1_delta = 0.0
    rho_2_delta = 0.0
    
    tsteps_2 = int((Tf-Tp)/dt) + 1
    transient_steps_2 = int(0.9 * tsteps_2)
    non_transient_steps_2 = tsteps_2 - transient_steps_2

    for i in range(transient_steps_2):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
    for i in range(non_transient_steps_2):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
        
        theta_exp = cp.exp(1j * theta)
        theta_exp_2 = cp.exp(1j * 2 * theta)

        rho_1_Tf += cp.abs(cp.sum(theta_exp) / N)  
        rho_2_Tf += cp.abs(cp.sum(theta_exp_2) / N)
    rho_1_Tf /= non_transient_steps_2
    rho_2_Tf /= non_transient_steps_2
        
    r1 = rho_1_Tf - rho_1_Tp
    r2 = rho_2_Tf - rho_2_Tp
    rho_1_delta = r1
    rho_2_delta = r2
    
    return cp.asnumpy(rho_1_delta), cp.asnumpy(rho_2_delta)

if __name__ == "__main__":
    # parameters of the model
    N = 10000
    K = 4.0
    L = 8.0
    
    # simulation parameters
    Tp = 1000
    Tf = 2000
    dt = 0.1

    
    # initial phase and frequency arrays
    cp.random.seed(12345)  # Use CuPy random generator
    omega_in = cp.random.standard_cauchy(N)
    theta_in = cp.random.uniform(-cp.pi, cp.pi, N)
    
    alpha_values = [0.001, 1]

    delta_rho_1 = cp.zeros(len(alpha_values))
    delta_rho_2 = cp.zeros(len(alpha_values))

    data = {"alpha_values": alpha_values, "delta_rho_1": [], "delta_rho_2": []}

    start_time = time.time()

    for i, alpha in enumerate(alpha_values):
        delta_rho_1[i], delta_rho_2[i] = evolve_and_perturb(
            theta_in.copy(), omega_in.copy(), K, L, N, Tp, dt, Tf, alpha
        )
        data["delta_rho_1"].append(float(delta_rho_1[i]))
        data["delta_rho_2"].append(float(delta_rho_2[i]))

    output_filename = "deltarhovsalphaK4L8.json"

    with open(output_filename, "w") as f:
        json.dump(data, f)

    end_time = time.time()
    print("GPU computation took", end_time - start_time, "seconds")

    print("Saved file as:", output_filename)
