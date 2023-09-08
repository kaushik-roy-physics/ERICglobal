import cupy as cp
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (5, 5)

# Load data from the JSON file
filename = "daidonormvsKLsmall.json"

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
plt.legend(fontsize = 16)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.savefig(f'{filename_prefix}_N10000_Gauss_rand.pdf')
plt.show()
