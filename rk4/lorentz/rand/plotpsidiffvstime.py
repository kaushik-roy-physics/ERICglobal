import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (4, 4)


# Function to generate the plots for multiple L values
def plot_psidiff_vs_time_multi_L(psis, L_values, T, tsteps, filename_prefix):
    t = np.linspace(0, T, tsteps)  
    
    colors = plt.get_cmap("tab10").colors

    for i, L in enumerate(L_values):
        psi_diff = psis[i]
        plt.plot(t, psi_diff, color=colors[i], label=f'$\Lambda$ = {L}')
        

    plt.xlabel('t', fontsize=18)
    plt.ylabel(r"$\Delta \Psi(t) = \Psi_2(t)- 2 \Psi_1(t)$", fontsize=18)
    
    plt.xticks(fontsize=14)  # Set x-axis tick fontsize
    plt.yticks(fontsize=16)  # Set y-axis tick fontsize

    plt.legend(loc='upper right', fontsize = 15)

    plt.savefig(f'{filename_prefix}_N10000_rand1.pdf')
    plt.show()

# Function to load the data from a JSON file
def load_data_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    
    L_values = data["L_values"]
    psis = [np.array(item["psi_diff"]) for item in data["psis"]]
    
    return psis, L_values


T = 1000
dt = 0.1
tsteps = int(T/dt)

filename = "psidiffvstimeK4Lhigh.json"
psis, L_values = load_data_from_json(filename)
filename_prefix = filename.split(".")[0]  # Extract the filename without the extension
plot_psidiff_vs_time_multi_L(psis, L_values, T, tsteps, filename_prefix)