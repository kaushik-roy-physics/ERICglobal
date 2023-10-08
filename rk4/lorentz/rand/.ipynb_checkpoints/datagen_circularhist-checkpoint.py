import cupy as cp
import json
import time

cp.random.seed(12345)

def dtheta_dt(theta, omega, K, L, N):
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = cp.sin(theta - theta[:, None]) ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt

N = 10000
T = 1000
dt = 0.1
tsteps = int(T / dt)

K = 4.0
L = 8.0

omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

theta = theta_in.copy()  # Initialize theta using theta_in

start_time = time.time()

for i in range(tsteps):
    theta += dtheta_dt(theta, omega_in, K, L, N) * dt  # Update theta in place
    theta = cp.mod(theta + cp.pi, 2 * cp.pi) - cp.pi

bins_theta = 100
theta_bins = cp.linspace(-cp.pi, cp.pi, bins_theta + 1)
hist, _ = cp.histogram(theta, bins=theta_bins)
hist = hist / N
hist = cp.roll(hist, shift=-bins_theta // 2)

theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2

data = {"theta_mid": cp.asnumpy(theta_mid).tolist(),
                   "hist": cp.asnumpy(hist).tolist()}

# Save data to a JSON file
output_filename = "circularhistogramK4L8.json"

with open(output_filename, 'w') as f:
    json.dump(data, f)

end_time = time.time()
print("GPU computation took:", end_time - start_time, "seconds")

print("Data saved to", output_filename)