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
N = 10000  # number of oscillators
K = 4.0  # coupling strength
L_values = 8.0  # relative strengths



# define the simulation parameters
T = 1000  # Integration time
dt = 0.1  # Timestep


# initialize the phase and natural frequency arrays
gamma = 0.5
omega_in = gamma * cp.random.standard_cauchy(N)
#omega_in = cp.random.standard_cauchy(N)
theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

start_time = time.time()

# Simulate the oscillator dynamics
final_theta, H_theta_final, H_derivative_final = calculate_quantities(theta_in, omega_in, K, L_values, N, T, dt)

#omega_avg = (1/N) * cp.sum(dtheta_dt(final_theta, omega_in, K,L_values,N) )
#print("Frequency of entrainment ", omega_avg)

# Calculate H_daido and its derivative for the final theta
H_daido_values = H_theta_final
H_derivative_values = H_derivative_final

# Find oscillators that satisfy both conditions

#condition1 = cp.abs(omega_in  - omega_avg -  (K * (H_daido_values - L_values/2))) < 1e-2

#omega_matrix = omega_in - omega_avg -  K * (H_daido_values - L_values/2)
#print(omega_matrix[3])

#condition1 = cp.isclose(omega_in, K * (H_daido_values - L_values / 2), atol = 0.01)

condition2 = H_derivative_values > 0


#indices_satisfying_conditions = cp.where(condition1 & condition2)

indices_satisfying_conditions = cp.where(condition2)[0]

omega_satisfying = omega_in[indices_satisfying_conditions].get()
theta_satisfying = final_theta[indices_satisfying_conditions].get()

num_stable_phases = len(omega_satisfying)
print("Number of stable phases:", num_stable_phases)

######################################

# Define a threshold for considering frequencies as almost the same
frequency_threshold = 1e-3

# Count of phases with almost the same frequencies
count_almost_same_frequencies = 0

# Iterate through the list of frequencies that satisfy condition2
for i in range(len(omega_satisfying)):
    freq_i = omega_satisfying[i]
    
    # Initialize a flag to check if the frequency is almost the same as others
    is_almost_same = False
    
    # Compare freq_i with other frequencies within the threshold
    for j in range(i+1, len(omega_satisfying)):
        freq_j = omega_satisfying[j]
        
        # Check if the absolute difference between freq_i and freq_j is within the threshold
        if abs(freq_i - freq_j) < frequency_threshold:
            is_almost_same = True
            break
    
    # If freq_i is almost the same as any other frequency, increment the count
    if is_almost_same:
        count_almost_same_frequencies += 1

# Print the count of phases with almost the same frequencies
print("Number of phases with almost the same frequencies:", count_almost_same_frequencies)

############################################################

############################################################

# Create an empty list to store frequencies of phases with almost identical frequencies
almost_same_frequencies = []

# Iterate through the list of frequencies that satisfy condition2
for i in range(len(omega_satisfying)):
    freq_i = omega_satisfying[i]
    
    # Initialize a flag to check if the frequency is almost the same as others
    is_almost_same = False
    
    # Compare freq_i with other frequencies within the threshold
    for j in range(i+1, len(omega_satisfying)):
        freq_j = omega_satisfying[j]
        
        # Check if the absolute difference between freq_i and freq_j is within the threshold
        if abs(freq_i - freq_j) < frequency_threshold:
            is_almost_same = True
            break
    
    # If freq_i is almost the same as any other frequency, add it to the list
    if is_almost_same:
        almost_same_frequencies.append(freq_i)

# Find the minimum and maximum frequencies among the almost same frequencies
if almost_same_frequencies:
    min_frequency = min(almost_same_frequencies)
    max_frequency = max(almost_same_frequencies)
else:
    min_frequency = None
    max_frequency = None

# Print the minimum and maximum frequencies
print("Minimum of frequencies:", min_frequency)
print("Maximum of frequencies:", max_frequency)

##################################################################

# Use bitwise_and to check both conditions simultaneously
#satisfying_oscillators = cp.bitwise_and(condition1, condition2)

# Extract omega and theta values for the oscillators satisfying the conditions using .get()
#omega_satisfying = omega_in[satisfying_oscillators].get()
#theta_satisfying = final_theta[satisfying_oscillators].get()

# Extract all omega_in and final_theta values for plotting using .get()
all_omega = omega_in.get()
all_theta = final_theta.get()

end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")

# Plot the results
plt.figure(figsize=(10, 6))

# Plot all the omega_in vs final_theta
plt.scatter(all_theta, all_omega, s=10, label='All Phases', alpha=0.5)

# Overlay the satisfying omega_in vs final_theta in a different color
plt.scatter(theta_satisfying, omega_satisfying, s=12, color='red', label='Stable phases')



# Shade the region between the minimum and maximum frequencies
if min_frequency is not None and max_frequency is not None:
    plt.axhspan(min_frequency, max_frequency, color='gray', alpha=0.3, label='Overlap')
    

plt.xlabel(r'$\theta_{i}$', fontsize = 20)
plt.ylabel(r'$\omega_{i}$', fontsize = 20)
plt.ylim(-20,4)
plt.title(r'$\omega_{i}$ vs $\theta_{i}$ at $t=1000$')
plt.legend(loc = 'lower right', fontsize = 15)
#plt.grid(True)
plt.savefig("omegavstheta_bistable_K4L8_Cauchy_gammahalf_t1000.pdf")
plt.show()

