# ERICmodel_global


This repository contains the Python scripts for generating the plots for the various physical quantities in the model proposed in 

We have provided a Jupyter Notebook that contains a simple implementation of the codes used to generate the plots in our model. It also contains useful text to help the readers get a context of the underlying theory. The .py files are the ones we used to generate the plots that appear in the manuscript.

The primary observables are macroscopic quantities such as the moduli of the order parameters z_1(t) and z_2(t) and their phases as a function of time for a particular choice
of K and \Lambda. Similarly we also plot the dependence of the stationary order parameter moduli \rho_1 and \rho_2 as a function of K and \Lambda. In addition , we also provide codes for plotting the phase maps of oscillator populations as a function of K and \Lambda. 

The time integration method used in the paper is RK4 method but we have also provided the code for Euler method of time integration to demonstrate that there is not much qualitative difference between the two methods.

One can play with the initial conditions for phases and frequencies as well as the coupling constants to see the behavior of the various quantities with changing parameters.

We have heavily relied on broadcasting capabilities of NumPy to make the code efficient and fast. Wherever applicable, we have used multiprocessing to reduce compilation time. 

