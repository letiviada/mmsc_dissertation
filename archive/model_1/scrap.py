import numpy as np
from get_functions import get_k
from casadi import *
def initial(nx=1, value=1):
    # Initialize the tensor with zeros (4, 4, 3, 3) and then replicate it nx times
    tensor = np.zeros((4, 4, 3, 3))

    # Populate G array using the dictionary
    G_dict = {
        (0, 2, 1, 2): 1, (2, 0, 1, 0): 1, 
        (1, 3, 1, 2): 2, (3, 1, 1, 0): 2, 
        (0, 1, 0, 1): 3, (1, 0, 2, 1): 3, 
        (2, 3, 0, 1): 4, (3, 2, 2, 1): 4, 
        (0, 1, 1, 1): 5, (1, 0, 1, 1): 5, 
        (0, 2, 1, 1): 6, (2, 0, 1, 1): 6, 
        (1, 3, 1, 1): 7, (3, 1, 1, 1): 7, 
        (2, 3, 1, 1): 8, (3, 2, 1, 1): 8
    }

    # Assign non-zero values to the tensor
    for key, val in G_dict.items():
        tensor[key] = val

    # Replicate the tensor nx times along a new axis
    initial_condition = np.tile(tensor, (nx, 1, 1, 1, 1))

    # Flatten the tensor to shape (nx * 4 * 4 * 3 * 3, 1)
    in_cond = initial_condition.reshape(-1, 1)
    
    return initial_condition, in_cond

nx = 10
_, G = initial(nx=10)
D = SX.ones((nx*144,1))
k = get_k(G,D,nx)
print(np.sum(G))
print(k)