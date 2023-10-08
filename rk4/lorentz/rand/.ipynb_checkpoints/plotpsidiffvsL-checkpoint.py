import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (5, 4)

def generate_plot_from_json(filename, filename_prefix):
    with open(filename, "r") as f:
        data = json.load(f)
    
    L_values = np.array(data["L_values"])
    psi_diff = np.array(data["psi_diff"])
    
    plt.plot(L_values, psi_diff)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel(r'$\Lambda$', fontsize=18)
    plt.ylabel(r'$\Delta \Psi = \Psi_{2} - 2\Psi_{1}$', fontsize = 18)
    
    plt.xticks(fontsize=16)  # Set x-axis tick fontsize
    plt.yticks(fontsize=16)  # Set y-axis tick fontsize
    
    plt.savefig(f'{filename_prefix}_rk4_N10000_Cauchy_rand.pdf')
    plt.ylim(-np.pi, np.pi)
    plt.show()

# Specify the filename of the JSON output
filename = "psidiffvsLK4.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
generate_plot_from_json(filename, filename_prefix)