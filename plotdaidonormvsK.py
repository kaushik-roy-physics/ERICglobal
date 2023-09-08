import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (5, 5)

# Load data from the JSON file
filename = "daidonormnearKLmedium.json"
#filename = "daidonormvsKchangingL.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

with open(filename, 'r') as f:
    loaded_data = json.load(f)

K_cpu = loaded_data["K_cpu"]
H_norm_values = loaded_data["H_norm_values"]
L_values = loaded_data["L_values"]

# Define colors based on the number of L_values
colors = plt.get_cmap("tab10").colors

for i, L in enumerate(L_values):
    plt.plot(K_cpu, H_norm_values[i], label=f"$\Lambda$ = {L}", color=colors[i])

plt.xlabel(r'$K$', fontsize=20)
plt.ylabel(r'$\vert \vert H \vert \vert$', fontsize=20)
plt.legend(loc = 'upper left', fontsize = 15)

# Set the number of x-ticks
#num_ticks = 5
#plt.xticks(K_cpu[::len(K_cpu)//num_ticks], fontsize=18)
#plt.xticks(fontsize = 18)

# Round off the numbers on the x-axis to two decimal places
#rounded_ticks = [f"{val:.2f}" for val in np.around(K_cpu, decimals=2)]


# Use MaxNLocator to determine the maximum number of x-ticks
locator = ticker.MaxNLocator(nbins=5)  # Adjust nbins as needed
plt.gca().xaxis.set_major_locator(locator)

# Round off the tick values to two decimal places
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
plt.show()
