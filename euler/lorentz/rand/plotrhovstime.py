import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (9, 5)

# Function to generate the plots for multiple L values
def plot_rho_vs_time_multi_L(rhos, L_values, T, tsteps, filename_prefix):
    t = np.linspace(0, T, tsteps) 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout(pad=2.0)
    
    colors = plt.get_cmap("tab10").colors

    for i, L in enumerate(L_values):
        rho_1, rho_2 = rhos[i]
        ax1.plot(t, rho_1, color=colors[i], label=f'$\Lambda$ = {L}')
        ax2.plot(t, rho_2, color=colors[i], label=f'$\Lambda$ = {L}')

    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel(r"${\rho_1(t)}$", fontsize=20)
    ax2.set_xlabel('t', fontsize=20)
    ax2.set_ylabel(r"${\rho_2(t)}$", fontsize=20)
    
    # Change the fontsize of the x-ticks and y-ticks
    ax1.tick_params(axis='both', which='both', labelsize=18)
    ax2.tick_params(axis='both', which='both', labelsize=18)
    
    # Adjust maximum number of y axis ticks
#    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))    
#    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
    
    # Adjust maximum number of x axis ticks
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))

    ax1.legend(loc='lower right', fontsize = 15)
    ax2.legend(loc='upper right', fontsize = 15)

    plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
    plt.show()
    
# Function to load the data from a JSON file
def load_data_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    L_values = data["L_values"]
    rhos = [(np.array(rho_data["rho_1"]), np.array(rho_data["rho_2"])) for rho_data in data["rhos"]]
    return rhos, L_values

T = 1000
dt = 0.1
tsteps = int(T/dt)

filename = "rhovstimeK3Lmedium.json"
rhos_loaded, L_values_loaded = load_data_from_json(filename)
filename_prefix = filename.split(".")[0]  # Extract the filename without the extension
plot_rho_vs_time_multi_L(rhos_loaded, L_values_loaded, T, tsteps, filename_prefix)
