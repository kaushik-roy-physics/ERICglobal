import cupy as cp
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (10,6)

filename = "bistable_phases_vs_initconds_K4L8.json"

filename_prefix = filename.split(".")[0]


# Load the data from the JSON file
with open(filename, 'r') as f:
    loaded_data = json.load(f)

all_phases_to_plot = np.array(loaded_data['all_phases_to_plot'])
all_trial_numbers = np.array(loaded_data['all_trial_numbers'])


# Example code for generating a scatter plot with different markers
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x', 'D', '|', '_']

for i, (phase, trial) in enumerate(zip(all_phases_to_plot, all_trial_numbers)):
    marker = markers[trial % len(markers)]  # Cycle through the marker list
    plt.scatter(phase, trial, marker=marker, alpha=0.5)

plt.xlabel(r'$\theta_{i}$', fontsize=20)
plt.xlim(-np.pi, np.pi)
plt.ylabel('Trial Number', fontsize=20)
plt.title('Phases in OR for each trial')
plt.savefig(f'{filename_prefix}_N10000_Gauss_rand.pdf')
plt.show()
