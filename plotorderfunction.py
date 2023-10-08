import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (6, 4)


def plot_order_function(filename, filename_prefix):
    # Load the data from the JSON file
    with open(filename, 'r') as f:
        loaded_data = json.load(f)

    theta_osc = np.array(loaded_data['theta_osc'])
    H_theta = np.array(loaded_data['H_theta'])
    L_values = loaded_data['L_values']
    
    colors = plt.get_cmap("tab10").colors
    plt.figure()

    for j, L in enumerate(L_values):
        theta_np = theta_osc[j]
        H_daido = H_theta[j]
        plt.scatter(theta_np, H_daido, s=2, color=colors[j], label=f"$\Lambda$ = {L_values[j]}")

    plt.xlabel(r'$\theta_i$', fontsize = 20)
    plt.ylabel(r'$H(\theta_i)$', fontsize = 20)
    plt.legend(fontsize = 14)

    x_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    x_ticklabels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
    plt.xticks(x_ticks, x_ticklabels, fontsize = 16)
    
    
    # Adjust y-axis ticks
    y_ticks = np.linspace(min(H_theta.flatten()), max(H_theta.flatten()), 6)
    y_ticks_rounded = [round(val, 1) for val in y_ticks]  # Round to 1 decimal place
    plt.yticks(y_ticks, y_ticks_rounded, fontsize=16)

    plt.savefig(f'{filename_prefix}_N10000_rand.pdf')
    plt.show()

# Specify the filename of the JSON output
filename = "daidoorderfunctionK10Lmedium.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
plot_order_function(filename, filename_prefix)
