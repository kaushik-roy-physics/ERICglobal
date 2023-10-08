import cupy as cp
import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (8,5)

filename = "omegavstheta_K4L1_t1000.json"

filename_prefix = filename.split(".")[0]

# Load the data from the JSON file
with open(filename, 'r') as f:
    loaded_data = json.load(f)

final_theta = np.array(loaded_data['final_theta'])
H_theta_final = np.array(loaded_data['H_theta_final'])
H_derivative_final = np.array(loaded_data['H_derivative_final'])
omega_satisfying = np.array(loaded_data['omega_satisfying'])
theta_satisfying = np.array(loaded_data['theta_satisfying'])
all_omega = np.array(loaded_data['all_omega'])
all_theta = np.array(loaded_data['all_theta'])


# Plot all the omega_in vs final_theta
plt.scatter(all_theta, all_omega, s=10, label='All Phases', alpha=0.5)

# Overlay the satisfying omega_in vs final_theta in a different color
plt.scatter(theta_satisfying, omega_satisfying, s=12, color='red', label='Stable locked phases')

plt.xlabel(r'$\theta_{i}$', fontsize=20)
plt.ylabel(r'$\omega_{i}$', fontsize=20)
plt.ylim(-4, 3)
plt.title(r'$\omega_{i}$ vs $\theta_{i}$ at $t=1000$')
plt.legend(loc='upper right', fontsize=15)
plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
plt.show()
