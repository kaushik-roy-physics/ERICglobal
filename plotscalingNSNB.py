import cupy as cp
import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (6,5)

filename = "scaling_bistableregions_K4L8.json"

filename_prefix = filename.split(".")[0]

# Code for importing data from the JSON file
with open(filename, "r") as f:
    saved_data = json.load(f)

N_values = saved_data["N_values"]
num_stable_phases_list = saved_data["num_stable_phases_list"]
count_OR_list = saved_data["count_OR_list"]

# Code for generating the plot
plt.scatter(N_values, num_stable_phases_list, s=10, marker='o', color='b', label=r'$N_{sl}$')
plt.xlabel('N', fontsize=20)
plt.scatter(N_values, count_OR_list, s=10, marker='^', color='g', label=r'$N_{bi}$')
plt.ylabel(r'$N_{sl},N_{bi}$', fontsize=20)
plt.legend(fontsize=15)
plt.title(r'Scaling of $N_{sl}$ and $N_{bi}$ with $N$')
plt.savefig(f'{filename_prefix}_Gauss_rand.pdf')
plt.show()
