from casadi import * 
import numpy as np

def interp_functions(k,j,tau_eval):
    k_inv = np.where(k != 0, 1/k, 0) # Obtain inverse of k
    interp_k = interpolant('INTERP_K','linear',[tau_eval],k)
    interp_k_inv = interpolant('INTERP_K_INV','linear',[tau_eval],k_inv)
    interp_j = interpolant('INTERP_J','linear',[tau_eval],j)
    return interp_k, interp_k_inv, interp_j