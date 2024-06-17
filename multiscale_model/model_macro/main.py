import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from solver import solv, run
from utils.get_functions import reshape_f
from utils.interpolate_functions import interp_functions
from utils.load import load_k_j
from plotting.plots_for_outputs import plot_time, plot_one_dim
from plotting.save import save_figure

def main():
    # Retrieve data from the microscale system
    #k_fun = lambda x: x**2 +1
    #k = k_fun(np.linspace(0,10,11))
    #k  = 2*np.ones(21)
    #j = 16*np.ones(21)
    k,j,tau_eval = load_k_j()
    #tau_eval = np.linspace(0,15,21)
    # Interpolate functions
    interp_k,interp_k_inv,interp_j = interp_functions(k,j,tau_eval)

    # Time Domain
    nt = 11
    t_eval = np.linspace(0,450,nt)
    print(t_eval)
    # Spatial Domain
    nx = 51 # If i put nx = 297 it stops working.
    x_eval = np.linspace(0,1,nx)

    #x_res,z_res = solv(interp_k,interp_k_inv,interp_j,t_eval,nx)
    F = solv(interp_k,interp_k_inv,interp_j,t_eval,nx)
    x_res, z_res = run(F,nx)
    c,tau,u,psi = reshape_f(x_res,z_res,nt,nx)
    print(type(psi))
    fig = plot_time(t_eval,x_eval,tau)
    save_figure(fig,'multiscale_model/figures/tau/tau')
    fig2 = plot_one_dim(tau_eval,k)
    save_figure(fig2,'multiscale_model/figures/permeability/k')

if __name__ == '__main__':
    main()