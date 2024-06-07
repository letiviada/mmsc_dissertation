import numpy as np
from casadi import *
from functools import reduce
from model import alg, ode

def shape_to_length(shape):
    """
    Set up the CasADi integrator for solving the DAE system.
    
    Parameters:
    shape (tuple): shape of the cell (i,j,r,s)
    
    Returns:
    size (int): returns the lenth a 1d array has to have: i*j*r*s
    """
    size= reduce (lambda x,y: x*y,shape)
    return size
     
def solver_dae(cell_shape, x_eval, t_eval):
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
    cell_size = shape_to_length(cell_shape)
    nx = len(x_eval)
    size_arrays = nx*cell_size
    x = SX.sym('x', (size_arrays, 1))
    z = SX.sym('z', (size_arrays, 1))
    
    opts = {'reltol': 1e-10, 'abstol': 1e-10}
    dae = {'x': x, 'z': z, 'ode': ode(x, z), 'alg': alg(x, z)}
    F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts)
    return F

def run(F,x0,z0):
    """
    Run the CasADi integrator given initial conditions.

    Parameters:
    F (Function): CasADi integrator function.
    x0 (np.ndarray): Initial conditions for state variables.
    z0 (np.ndarray): Initial conditions for algebraic variables.
    """
    result = F(x0=x0, z0=z0)
    x_res = result['xf'].full()
    z_res = result['zf'].full()

    return x_res, z_res

def reshape_N4(shape, cas_output):
    """
    Run the integrator to solve the DAE system.
    
    Parameters:
    shape (tuple): Shape of the periodic network.
    cas_output (list): List to reshape given as a nested list
    
    Returns:
    X (np.ndarray): tensor with the required size to interpret the solution.
    """
    X = cas_output.transpose().reshape(*shape)
    return X


