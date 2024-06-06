import  numpy as np

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

def initial(x_axis,t=0):
    """
    Initialize the tensor with given dimensions and time parameter.
    
    Parameters:
    x_axis (np.ndarray): Spatial points.
    t (float): Time parameter.
    
    Returns:
    tensor (np.ndarray): Initialised tensor.
    in_cond (np.ndarray): Flattened initial condition tensor.
    """
    nx = len(x_axis)
    tensor = np.zeros((nx,9, 4, 4)) # Shape of the tensor (r,i,j)
    def func(val,xi=0):
        c = 2 / val
        return 2 / (t + xi + c)
    values_dict = {
    (1, 0, 2): 1, (1, 1, 3): 2,
    (3, 0, 1): 3, (3, 2, 3): 4,
    (4, 0, 1): 5, (4, 0, 2): 6, (4, 1, 0): 5, (4, 1, 3): 7, (4, 2, 0): 6, (4, 2, 3): 8, (4, 3, 1): 7, (4, 3, 2): 8,
    (5, 1, 0): 9, (5, 3, 2): 10,
    (7, 2, 0): 11, (7, 3, 1): 12
    }
    # Assign non-zero values to the tensor
    positions = np.array(list(values_dict.keys()))
    values = np.array(list(values_dict.values()))
    idx_nx = np.arange(nx)[:, None]
    idx_r = positions[:, 0]
    idx_i = positions[:, 1]
    idx_j = positions[:, 2]
    xi = x_axis[:, None]
    tensor_values = func(values,xi)

    tensor[idx_nx, idx_r, idx_i, idx_j] = tensor_values

    #for i, xi in enumerate(x_axis):
       # for pos, val in values_dict.items():
           # tensor[i, pos[0], pos[1], pos[2]] = func(val, xi =xi)

    in_cond = tensor.reshape(-1)
    return tensor, in_cond
