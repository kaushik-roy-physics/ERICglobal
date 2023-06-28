# Code to plot the phases of the order parameters of the model
# as a function of time. The time integration method used is the
# RK4 method with time steps of dt = 0.1
#
# The line_profiler can be used to check
# which part of the program takes the most time to run in case
# someone ones to try a different implementation of the code.
# Run the following on the terminal:  kernprof -l script_name.py
# To see the results: python -m line_profiler script_name.py.lprof
# Replace the script_name with your own Python filename
#
#
# Author: Kaushik Roy
#
#
################################################################


import numpy as np
import matplotlib.pyplot as plt
import cmath


# import line_profiler

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (7,4)

# Computing dtheta/dt or the r.h.s of the phase equation
# We will use vectorization to update the entire r.h.s at each time step
# Please note that one can also use: 1/N * np.sum(np.sin((np.subtract.outer(theta, theta)), axis=0)) to achieve
# the same results

# @profile
def dtheta_dt(theta, omega, K,L):
    """Right Hand Side of dtheta/dt = ..."""
    sin_theta = np.sin(theta - theta[:, None])
    sinsq_theta = np.sin(theta - theta[:, None])**2

    sin_theta_sum = np.sum(sin_theta, axis=1)
    sinsq_theta_sum = np.sum(sinsq_theta, axis=1)

    dtheta_dt = np.add( omega, (1/N) * K * sin_theta_sum, (1/N) * K * L * sinsq_theta_sum)
    return dtheta_dt

# Updating the theta using RK4 method

# @profile
def rk4(theta,omega,K,L,dt):
    k1 = dt * dtheta_dt(theta, omega, K,L)
    k2 = dt * dtheta_dt(np.add(theta, 0.5 * k1), omega, K,L)
    k3 = dt * dtheta_dt(np.add(theta, 0.5 * k2), omega, K,L)
    k4 = dt * dtheta_dt(np.add(theta, k3), omega, K,L)
    theta = np.add(theta, np.multiply(1/6, np.add(np.add(np.add(k1, np.multiply(2, k2)), np.multiply(2, k3)), k4)))
    return theta


# Function to compute psi_1 and psi_2 as a function of time

# @profile
def generate_psi(theta, omega, K, L, dt, tsteps):
    """Function to return rho_1, rho_2"""
    psi_1 = np.zeros(tsteps)
    psi_2 = np.zeros(tsteps)
    
    for i in range(tsteps):
        theta = rk4(theta,omega,K,L,dt)
        r1 = np.mean(np.exp(1j*theta))
        r2 = np.mean(np.exp(1j*2*theta))
        psi_1[i] = cmath.phase(r1)
        psi_2[i] = cmath.phase(r2)
        
    return psi_1, psi_2

# Function to generate the plots

def plot_psi_vs_time(psi_1, psi_2, T, tsteps):
    t = np.linspace(0, T, tsteps)
    fig, (ax1, ax2) = plt.subplots(1, 2)
#    fig.suptitle('Order Parameters vs Time')
    fig.tight_layout(pad=2.0)
    ax1.plot(t, psi_1, color='b')
    ax2.plot(t, psi_2, color='c')
    ax1.set_xlabel('t', fontsize = 18)
    ax1.set_ylabel(r"${\Psi_{1}(t)}$", fontsize = 18)
    ax2.set_xlabel('t', fontsize = 18)
    ax2.set_ylabel(r"${\Psi_{2}(t)}$", fontsize = 18)
#    plt.savefig('psivstime_rk4.pdf')
    plt.show()


if __name__ == "__main__":
    # parameters of the model
    N = 2000
    K = 4.0
    L = 2.0
    
    # simulation parameters
    T = 1000
    dt = 0.1
    tsteps = int(T/dt)
    
    # initial phase and frequency arrays
    np.random.seed(12345)  # Legacy seeding algorithm for consistent random numbers
    omega = np.random.standard_cauchy(N)
    theta0 = np.random.uniform(-np.pi, np.pi, N)
    
    # generate psi
    psi_1, psi_2 = generate_psi(theta0, omega, K, L, dt, tsteps)
    
    plot_psi_vs_time(psi_1, psi_2, T, tsteps)