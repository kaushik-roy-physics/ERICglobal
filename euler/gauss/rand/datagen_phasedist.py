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

# define the Kuramoto model parameters
N = 10000  # number of oscillators
K = 3.0    # coupling strength
L = 6.0   # Relative coupling strength

# define the simulation parameters
T = 1000         # Integration time
dt = 0.1          # Timestep
tsteps = int(T/dt)  # total number of steps

# initialize the phase and natural frequency arrays
omega = cp.random.standard_normal(N)
theta = cp.random.uniform(-cp.pi, cp.pi, N)

theta_osc = cp.zeros(N)

start_time = time.time()

# simulate the Kuramoto model and compute the order parameter modulus
for i in range(tsteps):
    theta += dtheta_dt(theta, omega, K, L, N) * dt

theta_osc = (theta + cp.pi) % (2 * cp.pi) - cp.pi  # Ensuring angle is within -pi to pi


# Calculate the phase distribution as a function of theta
num_bins = 100  # Number of bins for the histogram
hist, bins = cp.histogram(theta_osc, bins=num_bins, range=(-cp.pi, cp.pi))

# Normalize the histogram to obtain the fraction (normalized distribution)
theta_centers = (bins[1:] + bins[:-1]) / 2
normalized_hist = hist / cp.sum(hist)

data = {"theta_centers": cp.asnumpy(theta_centers).tolist(),
                   "hist": cp.asnumpy(normalized_hist).tolist()}

# Save data to a JSON file
output_filename = "phasedistributionK3L6.json"

with open(output_filename, 'w') as f:
    json.dump(data, f)

end_time = time.time()
print("GPU computation took:", end_time - start_time, "seconds")

print("Data saved to", output_filename)
