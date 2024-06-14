import numpy as np
from casadi import *
from solver import solv, run
from utils.get_functions import reshape_f
from data.interpolate_functions import interp_functions
from data.utils import load_k_j

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
    #print(interp_k(4))
    # Time Domain
    nt = 11
    t_eval = np.linspace(0,10,nt)
    # Spatial Domain
    nx = 11 # If i put nx = 297 it stops working.
    x_eval = np.linspace(0,1,nx)

    #x_res,z_res = solv(interp_k,interp_k_inv,interp_j,t_eval,nx)
    F = solv(interp_k,interp_k_inv,interp_j,t_eval,nx)
    x_res, z_res = run(F,nx)
    c,tau,u,psi = reshape_f(x_res,z_res,nt,nx)
    print(c,tau,u,psi)

if __name__ == '__main__':
    main()