from functools import reduce

def map_indices(indices):
    """ Maps the keys as defined in a mathematical way to enable describing the tensor
    
    Parameters:
    indices (tuple): (i,j,r,s)
    
    Returns:
    (tuple): Mapped indices
    """
    i,j,r,s = indices
    key_i, key_j = i-1, j-1
    key_r, key_s = r+1, s+1
    return key_i, key_j, key_r, key_s

def create_value_dict(ic_dict):
    """
    Creates a dictionary with LaTeX formatted values.

    Parameters:
    ic_dict (dict): Dictionary with tuple keys (i, j, r, s) and values.

    Returns:
    dict: Dictionary with the same keys and LaTeX formatted values.
    """
    values_dict = {}
    for key in ic_dict.keys():
        i, j, r, s = key
        values_dict[key] = f"$G_{{{i}{j}}}^{{({r},{s})^T}}$"
    return values_dict

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


