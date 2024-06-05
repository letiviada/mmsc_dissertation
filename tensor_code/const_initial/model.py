import numpy as np

def alg(x,z):
    """ Definition of the algebraic part of the DAE """
    alg_eqn = z-0.5*x
    return alg_eqn

def ode(x,z):
    """ Definition of the differential right-hand side of the DAE"""
    ode = - x * z
    return ode

def initial(value = 1):
    tensor = np.zeros((9, 4, 4)) # Shape of the tensor (r,i,j)

    # Define initial values and their positions ()
    positions = [
        (1,0,2), (1,1,3),
        (3,0,1), (3,2,3),
        (4,0,1), (4,0,2), (4,1,0), (4,1,3), (4,2,0), (4,2,3), (4,3,1), (4,3,2),
        (5,1,0), (5,3,2),
        (7,2,0), (7,3,1)
    ]
    # Assign non-zero values to the tensor
    for pos in positions:
        tensor[pos] = value
    initial_condition  = tensor.reshape(-1) # Reshape into a 1d array (it does so by rows)
    return tensor, initial_condition
