import cupy as cp
import numpy as np
import json
import time
import matplotlib.pyplot as plt


cp.random.seed(12345)

def dtheta_dt(theta, omega, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    sinsq_theta = sin_theta ** 2

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    sinsq_theta_sum = cp.sum(sinsq_theta, axis=1)

    dtheta_dt = omega + ((1 / N) * K * sin_theta_sum) + ((1 / N) * K * L * sinsq_theta_sum)
    return dtheta_dt


def H_daido(theta, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = cp.sin(theta - theta[:, None])
    cos_2theta = cp.cos(2 * (theta - theta[:, None]))

    sin_theta_sum = cp.sum(sin_theta, axis=1)
    cos_2theta_sum = cp.sum(cos_2theta, axis=1)

    H_daido = - ((1 / N) * sin_theta_sum) + ((1 / N) * (L / 2) * cos_2theta_sum)
    
    return H_daido

def H_derivative_theta(theta, K, L, N):
    """Right Hand Side of dtheta/dt = ..."""
    cos_theta = cp.cos(theta - theta[:, None])
    sin_2theta = cp.sin(2 * (theta - theta[:, None]))

    cos_theta_sum = cp.sum(cos_theta, axis=1)
    sin_2theta_sum = cp.sum(sin_2theta, axis=1)

    H_derivative_theta = ( (1 / N) * cos_theta_sum ) + ( (1 / N) * L * sin_2theta_sum )
    
    return H_derivative_theta


def calculate_quantities(theta, omega, K, L ,N, T, dt):
    tsteps = int(T/dt)+1
    
    transient_steps = int(0.9 * tsteps)
    
    nontransient_steps = tsteps - transient_steps
   
    theta_osc = cp.zeros(N)
    H_theta = cp.zeros(N)
    H_derivative = cp.zeros(N)
    
    for t in range(transient_steps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
    for t in range(nontransient_steps):
        theta += dtheta_dt(theta, omega, K, L, N) * dt
        theta_osc = cp.mod(cp.unwrap(theta) + cp.pi, 2 * cp.pi) - cp.pi
        
        H_theta += H_daido(theta_osc, K, L, N)
        H_derivative += H_derivative_theta(theta_osc, K,L,N) 
    H_theta /= nontransient_steps
    H_derivative /= nontransient_steps
    
    return theta_osc, H_theta, H_derivative


# define the Kuramoto model parameters
N = 1000  # number of oscillators
K = 4.0  # coupling strength
L_values = 8.0  # relative strengths


# define the simulation parameters
T = 100  # Integration time
dt = 0.1  # Timestep


# initialize the phase and natural frequency arrays
#gamma = 0.05
#omega_in = gamma * cp.random.standard_cauchy(N)
omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

start_time = time.time()

# Simulate the oscillator dynamics
final_theta, H_theta_final, H_derivative_final = calculate_quantities(theta_in, omega_in, K, L_values, N, T, dt)

omega_avg = (1/N) * cp.sum(dtheta_dt(final_theta, omega_in, K,L_values,N) )

print("Frequency of entrainment ", omega_avg)


# Calculate H_daido and its derivative for the final theta
H_daido_values = H_theta_final
H_derivative_values = H_derivative_final

# Find oscillators that satisfy both conditions

omega_matrix = omega_in - omega_avg -  K * (H_daido_values - L_values/2)

dtheta_dt_avg = (1/N) * cp.sum(omega_matrix)

print("Avg of dtheta_dt:",dtheta_dt_avg) 

omega_matrix = omega_matrix.get()

#plt.hist(omega_matrix, bins=30, edgecolor="k", alpha=0.7)
#plt.title("Distribution of omega values")
#plt.xlabel("omega")
#plt.xlim(-40,40)
#plt.ylabel("Frequency")
#plt.grid(axis='y', alpha=0.75)
#plt.show()

plt.boxplot(omega_matrix)
plt.title("Boxplot of omega values")
plt.ylabel("omega")
plt.xticks([1], ['Oscillators'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
