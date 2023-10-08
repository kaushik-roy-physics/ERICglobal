import cupy as cp
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (9, 5)

def plot_data(L_cpu, rho_1_values, rho_2_values, K_values, filename_prefix):
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout(pad=2.0)
    
    colors = plt.get_cmap("tab10").colors

    for i, K in enumerate(K_values):
        ax[0].plot(L_cpu, rho_1_values[i], color=colors[i], label=f'$K$ = {K}')
        ax[1].plot(L_cpu, rho_2_values[i], color=colors[i], label=f'$K$ = {K}')

    ax[0].set_xlabel(r'$\Lambda$', fontsize=20)
    ax[0].set_ylabel(r"$\rho_1$", fontsize=20)
    ax[0].legend(loc='upper right', fontsize=14)
    
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    ax[1].set_xlabel(r'$\Lambda$', fontsize=20)
    ax[1].set_ylabel(r'$\rho_{2}$', fontsize=20)
    ax[1].legend(loc='upper center', fontsize=14)
    
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    # Change the fontsize of the x-ticks and y-ticks
    ax[0].tick_params(axis='both', which='both', labelsize=18)
    ax[1].tick_params(axis='both', which='both', labelsize=18)
    
    plt.savefig(f'{filename_prefix}_rk4_N10000_Cauchy_rand.pdf')
    
    plt.show()


filename = "rhovsLchangingK.json"

#filename = "rhovsLKmedium.json"
#filename = "rhovsLKhigh.json"

filename_prefix = filename.split(".")[0]  # Extract the filename without the extension

with open(filename, 'r') as f:
    data = json.load(f)

K_values = [item['K'] for item in data]
rho_1_values = [item['rho_1'] for item in data] 
rho_2_values = [item['rho_2'] for item in data]

L = cp.linspace(0, 5, 100)
L_cpu = L.get()  # To use for plotting

plot_data(L_cpu, rho_1_values, rho_2_values, K_values, filename_prefix)
