import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (5, 4)

def generate_plot_from_json(filename, filename_prefix):
    with open(filename, "r") as f:
        data = json.load(f)
    
    theta_centers = np.array(data["theta_centers"])
    normalized_hist = np.array(data["hist"])
    

    plt.plot(theta_centers, normalized_hist)
    plt.xlabel(r"$\theta$", fontsize = 20)
    plt.ylabel(r"$p(\theta)$", fontsize = 20)
    #plt.title("Phase Distribution of Kuramoto Oscillators")
    
    plt.xticks(fontsize=18)  # Set x-axis tick fontsize 
    
    # Adjust y-axis ticks
    y_ticks = np.linspace(min(normalized_hist.flatten()), max(normalized_hist.flatten()), 6)
    y_ticks_rounded = [round(val, 2) for val in y_ticks]  # Round to 1 decimal place
    plt.yticks(y_ticks, y_ticks_rounded, fontsize=18)
    
    plt.savefig(f'{filename_prefix}_N10000_Gauss_rand.pdf')
    plt.show()
    
# Specify the filename of the JSON output
filename = "phasedistributionK3L6.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
generate_plot_from_json(filename, filename_prefix)
