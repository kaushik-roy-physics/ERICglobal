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


start_time = time.time()

N = 10000

num_trials = 50

omega_in = cp.random.standard_normal((num_trials, N))
theta_in = cp.random.uniform(-cp.pi, cp.pi, size=(num_trials, N)) 

all_phases_to_plot = []
all_trial_numbers = []

for j in range(num_trials):
    omega_trial = omega_in[j, :]
    theta_trial = theta_in[j, :]
    
    K = 4.0
    L_values = 8.0
    T = 1000
    dt = 0.1

    final_theta, H_theta_final, H_derivative_final = calculate_quantities(theta_trial, omega_trial, K, L_values, N, T, dt)
    
    condition2 = H_derivative_final > 0
    indices_satisfying_conditions = cp.where(condition2)[0]

#    omega_satisfying = omega_in[indices_satisfying_conditions].get()
#    theta_satisfying = final_theta[indices_satisfying_conditions].get()

    omega_satisfying = omega_in[j][indices_satisfying_conditions].get()
    theta_satisfying = final_theta[indices_satisfying_conditions].get()

    frequency_threshold = 1e-3

    # Sort the array and get the indices that would sort it
    sorted_indices = cp.argsort(omega_satisfying)
    sorted_omega = omega_satisfying[sorted_indices]

    # Iterate over the sorted array
    for i in range(len(sorted_omega) - 1):  # -1 because we will check i with i+1
        if abs(sorted_omega[i] - sorted_omega[i + 1]) < frequency_threshold:
            all_phases_to_plot.append(theta_satisfying[sorted_indices[i]])
            all_trial_numbers.append(j + 1)



#    frequency_threshold = 1e-3
    
#    for i in range(len(omega_satisfying)):
#        freq_i = omega_satisfying[i]
#        is_almost_same = False
#        for k in range(i + 1, len(omega_satisfying)):
#            freq_k = omega_satisfying[k]
#            if abs(freq_i - freq_k) < frequency_threshold:
#                is_almost_same = True
#                break
#        if is_almost_same:
#            all_phases_to_plot.append(theta_satisfying[i])
#            all_trial_numbers.append(j + 1)

end_time = time.time()
print("GPU computation took", end_time - start_time, "seconds")



# Assuming all_phases_to_plot is populated as before
#all_phases_to_plot_np = np.array(all_phases_to_plot)  # Ensure it's a numpy array for plotting

# Create a histogram
#plt.figure(figsize=(10, 6))
#plt.hist(all_phases_to_plot_np, bins=50, edgecolor='black', alpha=0.7)  # Adjust 'bins' as needed

#plt.xlabel('Phase')
#plt.ylabel('Frequency')
#plt.title('Distribution of Phases across Trials')
#plt.xlim(-np.pi, np.pi)  # Assuming phases range from -pi to pi

#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.tight_layout()
#plt.show()


# Code to generate polar plots

# Assuming all_phases_to_plot and all_trial_numbers are populated as before
#all_phases_to_plot_np = np.array(all_phases_to_plot)  # Ensure it's a numpy array for plotting
#all_trial_numbers_np = np.array(all_trial_numbers)    # Ensure it's a numpy array for plotting

# Create a polar plot
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))

# Scatter plot on the polar axis
#ax.scatter(all_phases_to_plot_np, all_trial_numbers_np, marker='o', alpha=0.5)

# Set the direction of phase to increase clockwise
#ax.set_theta_direction(-1)

# Set 0 degree of phase to be on top
#ax.set_theta_offset(np.pi/2.0)

#ax.set_title("Phases across Trials", va='bottom')
#plt.show()

# Code for scatter plot with different markers

# A list of markers
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x', 'D', '|', '_']

plt.figure(figsize=(10, 6))
for i, (phase, trial) in enumerate(zip(all_phases_to_plot, all_trial_numbers)):
    marker = markers[trial % len(markers)]  # Cycle through the marker list
    plt.scatter(phase, trial, marker=marker, alpha=0.5)

plt.xlabel(r'$\theta_{i}$', fontsize = 20)
plt.xlim(-np.pi, np.pi)
plt.ylabel('Trial Number', fontsize = 20)
plt.title('Phases in OR for each trial')
plt.savefig('Phases_per_Trial.pdf')
plt.show()


# Code to generate simple scatter plot


#plt.figure(figsize=(10, 6))
#plt.scatter(all_phases_to_plot, all_trial_numbers, marker='o', alpha=0.5)
#plt.xlabel(r'$\theta_{i}$', fontsize = 20)
#plt.xlim(-np.pi, np.pi)
#plt.ylabel('Trial Number', fontsize = 20)
#plt.title('Bistable phases for each trial')
#plt.colorbar(label='Phase Value')
#plt.savefig('Phases_per_Trial.pdf')
#plt.show()
