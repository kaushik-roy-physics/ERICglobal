import cupy as cp
import json
import time

cp.random.seed(12345)

def dtheta_dt(theta, omega, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = cp.sin(theta - theta[:, None]) ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt


N = 10000
K = 4.0
L = 6.0

T = 1000
dt = 0.1
tsteps = int(T/dt)

omega = cp.random.standard_cauchy(N)
theta = cp.random.uniform(-cp.pi, cp.pi, N)

theta_osc = cp.zeros(N)

start_time = time.time()

data = {'theta_osc': []}

for i in range(tsteps):    
    theta += dtheta_dt(theta, omega, K, L, N) * dt

theta_osc = (theta + cp.pi) % (2 * cp.pi) - cp.pi
data['theta_osc'].append(cp.asnumpy(theta_osc).tolist())

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")

# Save data to a JSON file
output_filename = "phasesK4L6.json"

with open(output_filename, 'w') as f:
    json.dump(data, f)

print("Data saved to", output_filename)
