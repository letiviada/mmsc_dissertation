import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from solver import solv, run
from utils.get_functions import reshape_f
from utils.interpolate_functions import interp_functions
from utils.load import load_k_j
from multiscale_model.plotting.plots_for_outputs import plot_time, plot_one_dim
from multiscale_model.plotting.save import save_figure

def main():
    # Retrieve data from the microscale system
    l = 2.0
    k,j,tau_eval = load_k_j()
    #tau_eval = np.linspace(0,15,21)
    # Interpolate functions
    interp_k,interp_k_inv,interp_j = interp_functions(k,j,tau_eval)
 
    # Time Domain
    nt = 101
    t_eval = np.linspace(0,1,nt)
    # Spatial Domain
    nx = 101 # If i put nx = 297 it stops working.
    x_eval = np.linspace(0,l,nx)

    #x_res,z_res = solv(interp_k,interp_k_inv,interp_j,t_eval,nx)
    F = solv(interp_k,interp_k_inv,interp_j,t_eval,nx,l)
    x_res, z_res = run(F,nx)
    c,tau,u,psi = reshape_f(x_res,z_res,nt,nx)
    #fig = plot_time(t_eval,x_eval,tau)
    #save_figure(fig,'multiscale_model/figures/tau/tau')
    fig = plot_one_dim(t_eval,[psi[:,0]])
    save_figure(fig,'multiscale_model/figures/reactivity/psi')
    fig = plot_one_dim(t_eval,[u])
    save_figure(fig,'multiscale_model/figures/darcy_velocity/u')

if __name__ == '__main__':
    main()