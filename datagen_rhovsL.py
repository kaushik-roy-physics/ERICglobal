import cupy as cp
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

    return rho_1, rho_2


N = 10000
T = 1000
dt = 0.1
tsteps = int(T / dt)

L = cp.linspace(0, 3, 100)
L_cpu = L.get()  # To use for plotting
K_values = [1, 3, 4, 5]


omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

data_to_save = []

start_time = time.time()

for K in K_values:
    rho_1 = cp.zeros(len(L))
    rho_2 = cp.zeros(len(L))

    for i in cp.arange(len(L)):
        rho_1[i], rho_2[i] = calculate_order_parameters(K, dt, tsteps, omega_in, theta_in, L[i], N)

    data_to_save.append({
        'K': K,
        'rho_1': rho_1.get().tolist(),
        'rho_2': rho_2.get().tolist()
    })

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")

output_filename = "rhovsLK.json"

#output_filename = "rhovsLKmedium.json"
#output_filename = "rhovsLKhigh.json"

with open(output_filename, 'w') as f:
    json.dump(data_to_save, f)
    
print("Data saved to", output_filename)
