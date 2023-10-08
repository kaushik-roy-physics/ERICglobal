import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (5, 4)

def generate_plot_from_json(filename, filename_prefix):
    with open(filename, "r") as f:
        data = json.load(f)
    
    L_values = np.array(data["L_values"])
    phi_vals = np.array(data["phi_vals"])
    
    plt.plot(L_values, phi_vals)
    plt.axhline(np.arctan(4), color='black', linestyle='--', linewidth=1)
    plt.xlabel(r'$\Lambda$', fontsize=18)
    plt.ylabel(r'$\phi = \arctan{ (2 \rho_1 / \Lambda \rho_2) } $', fontsize=18)
    
    plt.xticks(fontsize=16)  # Set x-axis tick fontsize
    plt.yticks(fontsize=16)  # Set y-axis tick fontsize
    
    plt.savefig(f'{filename_prefix}_N10000_rand1.pdf')
    plt.ylim(0, np.pi/2)
    plt.show()

# Specify the filename of the JSON output
filename = "phivsLK3.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
generate_plot_from_json(filename, filename_prefix)
