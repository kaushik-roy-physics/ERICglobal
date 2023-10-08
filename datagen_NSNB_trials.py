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

omega_in = cp.random.standard_normal((num_trials,N))
theta_in = cp.random.uniform(-cp.pi, cp.pi,size=(num_trials,N)) 

num_stable_phases_list = []
count_OR_list = []

for j in range(num_trials):                # For each K, iterate over defined number of trials
    omega = omega_in[j,:]                   # Frequency sampling for jth trial and all N oscillators 
    theta = theta_in[j,:] 


    
    K = 4.0  # coupling strength
    L_values = 8.0  # relative strengths



    # define the simulation parameters
    T = 1000  # Integration time
    dt = 0.1  # Timestep


    # initialize the phase and natural frequency arrays
    #gamma = 0.5
    #omega_in = gamma * cp.random.standard_cauchy(N)
    #omega_in = cp.random.standard_normal(N)
    #theta_in = cp.random.uniform(-cp.pi, cp.pi, N)

    

    # Simulate the oscillator dynamics
    final_theta, H_theta_final, H_derivative_final = calculate_quantities(theta, omega, K, L_values, N, T, dt)

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

    omega_satisfying = omega_in[j][indices_satisfying_conditions].get()
    theta_satisfying = final_theta[indices_satisfying_conditions].get()

    num_stable_phases = len(omega_satisfying)

    num_stable_phases_list.append(num_stable_phases)
    
    # Calculate count_OR
    frequency_threshold = 1e-3
    count_OR = 0

    # Sort the omega_satisfying array
    omega_satisfying.sort()

    # Calculate the absolute differences between adjacent elements
    freq_diff = np.diff(omega_satisfying)

    # Check if any difference is below the threshold
    is_almost_same = freq_diff < frequency_threshold

    # Count occurrences where the difference is below the threshold
    count_OR = np.sum(is_almost_same)

    count_OR_list.append(count_OR)

    # Calculate count_OR
#    frequency_threshold = 1e-3
#    count_OR = 0

#    for i in range(len(omega_satisfying)):
#        freq_i = omega_satisfying[i]

#        is_almost_same = False

#        for j in range(i + 1, len(omega_satisfying)):
#            freq_j = omega_satisfying[j]
#
#            if abs(freq_i - freq_j) < frequency_threshold:
#                is_almost_same = True
#               break

#        if is_almost_same:
#            count_OR += 1

 #   count_OR_list.append(count_OR)
    
end_time = time.time()

print("GPU computation took", end_time - start_time, "seconds")


output_filename = "NSNB_vs_trials_K4L8.json"


# Save the data to a JSON file
data_to_save = {
    'num_stable_phases_list': num_stable_phases_list,
    'count_OR_list': count_OR_list,
}

with open(output_filename, 'w') as f:
    json.dump(data_to_save, f)
    
print("Data saved as:", output_filename)