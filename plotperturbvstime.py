import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import cupy as cp

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (9, 5)

# Function to generate the plots
def plot_rho_vs_time(delta_rho_1, delta_rho_2, T0, Tf, tsteps, alpha_values):
    t = cp.asnumpy(cp.linspace(T0, Tf, tsteps))  # Time after perturbation
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout(pad=2.0)

#    curve_alpha = 0.3  # Set transparency level for each curve

#    for i, alpha in enumerate(alpha_values):
#        ax1.plot(t, cp.asnumpy(delta_rho_1[i]), label=r'$\alpha$ = {:.3f}'.format(alpha), alpha=curve_alpha)  # Convert CuPy to NumPy here with added transparency
#        ax2.plot(t, cp.asnumpy(delta_rho_2[i]), label=r'$\alpha$ = {:.3f}'.format(alpha), alpha=curve_alpha)  # Convert CuPy to NumPy here with added transparency

#    ax1.set_xlabel('t', fontsize=20)
#    ax1.set_ylabel(r"$\Delta \rho_1(t)=\rho_{1}(t)-\rho_{1}(T_{p})$", fontsize=20)
#    ax2.set_xlabel('t', fontsize=20)
#    ax2.set_ylabel(r"$\Delta \rho_2(t)=\rho_{2} (t)-\rho_{2}(T_{p})   $", fontsize=20)

    # Change the fontsize of the x-ticks and y-ticks
#    ax1.tick_params(axis='both', which='both', labelsize=18)
#    ax2.tick_params(axis='both', which='both', labelsize=18)

#    # Adjust maximum number of x axis ticks
#    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
#    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    
#    ax1.legend(loc= 'upper right',fontsize = 14)
#    ax2.legend(loc = 'lower right', fontsize = 14)

#    plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
#    plt.show()

def plot_rho_vs_time(delta_rho_1, delta_rho_2, T0, Tf, tsteps, alpha_values):
    t = cp.asnumpy(cp.linspace(T0, Tf, tsteps))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout(pad=2.0)
    
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', 'D']
    for i, alpha in enumerate(alpha_values):
        ax1.plot(t, cp.asnumpy(delta_rho_1[i]), label=r'$\alpha$ = {:.3f}'.format(alpha), linestyle=line_styles[i % 4], marker=markers[i % 4], markevery=50)
        ax2.plot(t, cp.asnumpy(delta_rho_2[i]), label=r'$\alpha$ = {:.3f}'.format(alpha), linestyle=line_styles[i % 4], marker=markers[i % 4], markevery=50)
    
    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel(r"$\Delta \rho_1(t)=\rho_{1}(t)-\rho_{1}(T_{p})$", fontsize=20)
    ax2.set_xlabel('t', fontsize=20)
    ax2.set_ylabel(r"$\Delta \rho_2(t)=\rho_{2} (t)-\rho_{2}(T_{p})$", fontsize=20)

    ax1.tick_params(axis='both', which='both', labelsize=18)
    ax2.tick_params(axis='both', which='both', labelsize=18)

    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax1.legend(loc= 'upper right',fontsize = 14)
    ax2.legend(loc = 'upper right', fontsize = 14)

    plt.savefig(f'{filename_prefix}_N10000_Cauchy_rand.pdf')
    plt.show()


if __name__ == "__main__":

    filename = "deltarhovstimeK4L8.json"

    filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

    
    # Load data from the JSON file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract data
    delta_rho_1_list = [cp.array(item) for item in data['delta_rho_1_list']]
    delta_rho_2_list = [cp.array(item) for item in data['delta_rho_2_list']]
    T0 = data['T0']
    Tf = data['Tf']
    tsteps = data['tsteps']
    alpha_values = data['alpha_values']

    # Plot
    plot_rho_vs_time(delta_rho_1_list, delta_rho_2_list, T0, Tf, tsteps, alpha_values)
