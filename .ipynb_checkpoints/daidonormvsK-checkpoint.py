import cupy as cp
import matplotlib.pyplot as plt
import time

cp.random.seed(12345)

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (5, 5)

def dtheta_dt(theta, omega, K, L, N):
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt


def H_daido(theta, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    cos_2theta = cp.cos(2*(theta - theta[:, None]))

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    cos_2theta_sum = cp.sum(cos_2theta, axis=1)

    H_daido = - ( (1/N) * sin_theta_sum ) + ( (1/N) * (L/2) * cos_2theta_sum )
    return H_daido


# Compute Daido norm for a given K, L, N, T, dt, omega, and theta
def compute_daido_norm(K, L, N, T, dt, omega, theta):
    tsteps = int(T / dt)
    transient_steps = int(0.9 * tsteps)
    non_transient_steps = tsteps - transient_steps
    
    daido_norms = cp.array([])  # Initialize as a Cupy array

    for i in range(transient_steps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
    for i in range(non_transient_steps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt

        H = cp.sqrt(2 * (cp.pi/N) * cp.sum(cp.abs(H_daido(theta, K, L, N))**2))
        daido_norms = cp.append(daido_norms, H)  # Append to Cupy array

    return cp.mean(daido_norms)

N = 100
T = 300
dt = 0.1
tsteps = int(T / dt)

K = cp.linspace(0, 5, 100)
K_cpu = K.get()  # To use for plotting
L_value = 0.3  # Use a single L value

omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

H_norm_values = []

start_time = time.time()

H_norm = cp.zeros(len(K))

for i in cp.arange(len(K)):
    H_norm[i] = compute_daido_norm(K[i], L_value, N, T, dt, omega_in, theta_in)

H_norm_values.append(H_norm.get())

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")


plt.plot(K_cpu, H_norm_values[0], label=f"$\Lambda$ = {L_value}", color='b')

plt.xlabel(r'$K$', fontsize=18)
plt.ylabel(r'$\vert \vert H \vert \vert$', fontsize=18)
plt.legend()
#plt.savefig('Daido_Norm_vs_KLsmall_Cauchy_rand1.pdf')
plt.show()
