import numpy as np
from casadi import *
from model import alg, ode, initial

def solver_dae(array_length, x_eval, t_eval):
    """
    Set up the CasADi integrator for solving the DAE system.
    
    Parameters:
    array_length (int): Length of the array.
    x_eval (np.ndarray): Spatial points.
    t_eval (np.ndarray): Time evaluation points.
    
    Returns:
    Function: CasADi integrator function.
    """
    # Number of points
    nt = len(t_eval)
    nx = len(x_eval)
    size_arrays = nx*array_length
    x = SX.sym('x', (size_arrays, 1))
    z = SX.sym('z', (size_arrays, 1))
    
    opts = {'reltol': 1e-10, 'abstol': 1e-10}
    dae = {'x': x, 'z': z, 'ode': ode(x, z), 'alg': alg(x, z)}
    F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts)
    return F

def run_reshape_N4(F, x0, z0, nt, nx):
    """
    Run the integrator to solve the DAE system.
    
    Parameters:
    F (Function): CasADi integrator function.
    x0 (np.ndarray): Initial conditions for state variables.
    z0 (np.ndarray): Initial conditions for algebraic variables.
    nt (int): Number of time points.
    nx (int): Number of spatial points.
    
    Returns:
    tuple: Arrays of state and algebraic variables.
    """
    result = F(x0=x0, z0=z0)
    x_res = result['xf'].full()
    z_res = result['zf'].full()
    
    X = x_res.transpose().reshape(nt, nx, 9, 4, 4)
    Z = z_res.transpose().reshape(nt, nx, 9, 4, 4)
    return X, Z


