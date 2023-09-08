import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (6, 4)

def generate_plot_from_json(filename, filename_prefix):
    with open(filename, "r") as f:
        data = json.load(f)
    
    L_values = np.array(data["L_values"])
    rho_ratios = np.array(data["rho_ratios"])
    
    plt.plot(L_values, rho_ratios)
    plt.axhline(1, color='black', linestyle='--', linewidth=1)
    
    plt.xlabel(r'$\Lambda$', fontsize=18)
    plt.ylabel(r'$\rho_2 /  \rho_{1}^{2} $', fontsize=18)
    
    x_ticks = [0.0, 0.5, 1, 1.5, 2, 2.5, 3.0]
    plt.xticks(x_ticks, fontsize = 16)
               
    y_ticks = np.linspace(min(rho_ratios.flatten()), max(rho_ratios.flatten()), 6)
    y_ticks_rounded = [round(val, 1) for val in y_ticks]  # Round to 1 decimal place
    plt.yticks(y_ticks, y_ticks_rounded, fontsize=16)
    
    plt.savefig(f'{filename_prefix}_N10000_rand1.pdf')
    plt.ylim(0, np.pi/2)
    plt.show()

# Specify the filename of the JSON output
filename = "OAtestK4.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
generate_plot_from_json(filename, filename_prefix)