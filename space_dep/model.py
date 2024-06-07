import  numpy as np
from data.N4 import ic_dict
def alg(x,z):
    """
    Algebraic equation for the DAE system.
    
    Parameters:
    x (SX): State variable.
    z (SX): Algebraic variable.
    
    Returns:
    SX: Algebraic equation.
    """
    alg_eqn = z-0.5*x
    return alg_eqn

def ode(x,z):
    """
    Ordinary differential equation for the DAE system.
    
    Parameters:
    x (SX): State variable.
    z (SX): Algebraic variable.
    
    Returns:
    SX: ODE equation.
    """
    ode = - x * z
    return ode

def initial(x_eval,shape,t=0):
    """
    Initialize the tensor with given dimensions and time parameter.
    
    Parameters:
    x_eval (np.ndarray): Spatial points.
    shape (tuple): Shape of the cells (i,j,r,s) 
    t (float): Time parameter.
    
    Returns:
    tensor (np.ndarray): Initialised tensor.
    in_cond (np.ndarray): Flattened initial condition tensor.
    """
    nx = len(x_eval)
    tensor = np.zeros((nx,*shape)) #(xi,i,j,r,s): initial_conductance at edge i,j,r,s at position xi
    
    def func(val,xi=0):
        c = 2 / val
        return 2 / (t + xi + c)
    
    # Assign non-zero values to the tensor
    positions = np.array(list(ic_dict.keys()))
    values = np.array(list(ic_dict.values()))
    idx_nx = np.arange(nx)[:, None]
    idx_i, idx_j, idx_r, idx_s  = positions[:, 0], positions[:, 1], positions[:, 2], positions[:,3]
    xi = x_eval[:, None]
    tensor_values = func(values,xi)

    # Build tensor for the initial condition depending on space
    tensor[idx_nx, idx_i, idx_j, idx_r, idx_s] = tensor_values

    # Reshape to obtain solution in a form CasADi accepts
    in_cond = tensor.reshape(-1)
    return tensor, in_cond
