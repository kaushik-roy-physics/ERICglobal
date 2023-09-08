import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (6,4)

def plot_phases(theta_osc, filename_prefix):
    plt.scatter(range(len(theta_osc)), theta_osc, s=3)
    plt.xlabel('N', fontsize=20)
    plt.ylabel(r'$\theta_i$', fontsize=20)
    plt.ylim(-4, 4)

    y_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    y_ticklabels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
    plt.yticks(y_ticks, y_ticklabels, fontsize=18)
    plt.xticks(fontsize=16)
    
    plt.savefig(f'{filename_prefix}_N10000_Gauss_rand.pdf')
    plt.show()

# Load data from JSON file
filename = "phasesK3L6.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

with open(filename, 'r') as f:
    data = json.load(f)

theta_osc_list = data['theta_osc']

for theta_osc in theta_osc_list:
    plot_phases(theta_osc, filename_prefix)
