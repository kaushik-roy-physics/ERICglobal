import numpy as np
import matplotlib.pyplot as plt
import json

#plt.rcParams["figure.autolayout"] = True
#plt.rcParams["figure.figsize"] = (5, 4)

def generate_plot_from_json(filename, filename_prefix):
    with open(filename, "r") as f:
        data = json.load(f)
    
    theta_mid = np.array(data["theta_mid"])
    hist = np.array(data["hist"])
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    colors = hist / hist.max()
    cmap = plt.cm.viridis
    
    bins_theta = 100
    bars = ax.bar(theta_mid, hist, width=2 * np.pi / bins_theta, color=cmap(colors), edgecolor='k')

    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels(['0', '30', '60', '90', '120', '150', '180', '-150', '-120', '-90', '-60', '-30'], fontsize = 15)

    ax.set_yticklabels([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=ax, pad=0.1)
    colorbar.ax.tick_params(labelsize=14)

    plt.savefig(f'{filename_prefix}_N10000_Gauss_rand.pdf')
    plt.show()
    
# Specify the filename of the JSON output
filename = "circularhistogramK4L6.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

# Generate the plot from the JSON data
generate_plot_from_json(filename, filename_prefix)
