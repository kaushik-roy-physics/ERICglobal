import cupy as cp
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (8,6)

filename = "NSNB_vs_trials_K4L8.json"

filename_prefix = filename.split(".")[0]


# Load the data from the JSON file
with open(filename, 'r') as f:
    loaded_data = json.load(f)

num_stable_phases_list = loaded_data['num_stable_phases_list']
count_OR_list = loaded_data['count_OR_list']

trial_numbers = list(range(1, num_trials + 1))

plt.scatter(trial_numbers, num_stable_phases_list, s=10, marker='o', color='b', label=r'$N_{sl}$')
plt.scatter(trial_numbers, count_OR_list, s=10, marker='^', color='r', label=r'$N_{bi}$')

plt.xlabel('Trial Number', fontsize=20)
plt.ylabel(r'$Value$', fontsize=20)
plt.legend(fontsize=15)
plt.title(r'$N_{s}$ and $N_{OR}$ for different trials')
plt.savefig(f'{filename_prefix}_N10000_Gauss_rand.pdf')
plt.show()
