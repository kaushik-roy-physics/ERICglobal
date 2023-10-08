import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (4, 4)

# Define a function to plot delta_rho_1 and delta_rho_2 as a function of alpha
#def plot_rho_vs_alpha(delta_rho_1, delta_rho_2, alpha_values):
#    plt.figure()
#    plt.plot(alpha_values, delta_rho_1, label=r"$\Delta \rho_{1}$")
#    plt.plot(alpha_values, delta_rho_2, label=r"$\Delta \rho_{2}$")
#    plt.xlabel(r"$\alpha$", fontsize = 20)
#    plt.ylabel(r"$\Delta \rho_{1,2}$", fontsize = 20)
#    plt.xscale('log')  # Set x-axis to log scale
#    plt.xticks([1e-3, 1e-2, 1e-1, 1], ['1e-3', '1e-2', '1e-1', '1'], fontsize = 18)  # Custom x-axis ticks and labels
#    plt.yticks(fontsize = 18)
#    plt.ylim(-0.1,0.1)
#    plt.legend(fontsize = 15)
#    plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
#    plt.show()

# Define a function to plot delta_rho_1 and delta_rho_2 as a function of alpha
def plot_rho_vs_alpha(delta_rho_1, delta_rho_2, alpha_values):
    plt.figure()
    
    line_styles = ['-', '--']
    markers = ['o', '^']
    
    # Plot delta_rho_1 with a specific line style, marker, and color
    plt.plot(alpha_values, delta_rho_1, label=r"$\Delta \rho_{1}$", linestyle=line_styles[0], marker=markers[0], markevery = 2)
    
    # Plot delta_rho_2 with a different line style, marker, and color
    plt.plot(alpha_values, delta_rho_2, label=r"$\Delta \rho_{2}$", linestyle=line_styles[1], marker=markers[1], markevery = 2)
    
    plt.xlabel(r"$\alpha$", fontsize = 20)
    plt.ylabel(r"$\Delta \rho_{1,2}$", fontsize = 20)
    plt.xscale('log')  # Set x-axis to log scale
    plt.xticks([1e-3, 1e-2, 1e-1, 1], ['1e-3', '1e-2', '1e-1', '1'], fontsize = 18)  # Custom x-axis ticks and labels
    plt.yticks(fontsize = 18)
    plt.ylim(-0.1,0.1)
    plt.legend(fontsize = 15)
    plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
    plt.show()



if __name__ == "__main__":

    filename = "deltarhovsalphaK4L8.json"

    filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

    with open(filename, "r") as f:
        data = json.load(f)

    alpha_values = data["alpha_values"]
    delta_rho_1 = data["delta_rho_1"]
    delta_rho_2 = data["delta_rho_2"]

    plot_rho_vs_alpha(delta_rho_1, delta_rho_2, alpha_values)
