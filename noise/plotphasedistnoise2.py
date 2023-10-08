import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (7, 4)

def generate_plot_from_json(filename, filename_prefix):
    with open(filename, "r") as f:
        data_list = json.load(f)

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'H', 'X', '+']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']  # List of distinct linestyles

    for i, data in enumerate(data_list):
        D = data["D"]
        theta_centers = np.array(data["theta_centers"])
        normalized_hist = np.array(data["hist"])

        # Use a distinct marker and linestyle for each data set
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        plt.plot(theta_centers, normalized_hist,ms=5, label= rf'$D$ = {D:.1f}', marker=marker, markevery=5, linestyle=linestyle)

    plt.xlabel(r"$\theta$", fontsize = 20)
    plt.ylabel(r"$p(\theta)$", fontsize = 20)
#    plt.legend(loc = 'upper right', fontsize = 14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    
    plt.xticks(fontsize=18)  # Set x-axis tick fontsize 
    
    # Adjust y-axis ticks
    y_ticks = np.linspace(min(normalized_hist.flatten()), max(normalized_hist.flatten()), 6)
    y_ticks_rounded = [round(val, 2) for val in y_ticks]  # Round to 1 decimal place
    plt.yticks(y_ticks, y_ticks_rounded, fontsize=18)
    
    plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand_2.pdf', bbox_inches='tight')
    plt.show()
    
# Specify the filename of the JSON output
filename = "phasedistribution_K4L8_noise.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
generate_plot_from_json(filename, filename_prefix)
