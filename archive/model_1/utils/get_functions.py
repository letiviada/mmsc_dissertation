from utils.help_functions import integrate_inverse_k, inverse_k, sum_over_last_index
import numpy as np
from casadi import *
def get_r(nx):
    """
    Calculate the tensor r and reshape it to a 1D array.

    Parameters:
    nx (int): Number of spatial points.

    Returns:
    r_array (np.ndarray): 1D array containing the values of r to be used when defining k.
    """
        
    shape = (4, 4, 3, 3)
    r_tensor = np.zeros(shape)

    # Assign values based on r and s
    for r in range(-1, 2):
        for s in range(-1, 2):
            if (r, s) in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                value = 0
            else:
                value = r + s
            r_tensor[:, :, r + 1, s + 1] = value
           
    if nx== 1:
        r_tens = r_tensor
    else:
        r_tens = np.tile(r_tensor, (nx,1,1,1))

    r_array = r_tens.reshape(-1)
    return r_array

def get_k(G, Delta, nx):
    """
    Find k from summing over i, j, r and s

    Parameters:
    G :
    Delta:
    nx (int): Number of spatial points

    Returns:
    k
    """
    r_array = get_r(nx)
    r_array = np.ones((nx*144,1))
    result = sum_over_last_index(1* G * Delta, (nx,4,4,3,3))
    result2= sum_over_last_index(result,(nx,4,4,3))
    result3 = sum_over_last_index(result2,(nx,4,4))
    k = sum_over_last_index(result3,(nx,4))
    k = 0.5 * k
    k = SX.ones((nx,1))
    return k

def get_u(k,x_eval):
    """
    Find u(t) = 1/(\int^{l}_{0}(1/k(t,x))dx)

    Parameters:
    k (np.ndarray): Array containing the values of k(t,x) evaluated in the discretised spatial domain.
    x_eval (np.ndarray): Points used to discretise the domain.

    Returns:
    u (float): u is a constant in space so for each time point we only get one value.
    """

    u_inv = integrate_inverse_k(k=k,x_eval=x_eval, i = 0)
    #if u_inv == 0:
       # raise ValueError("u_inv cannot be 0")
    #else:
    u = 1 / u_inv
    return u

def get_p(k,x_eval):
    """
    Find p(t,x) = \int^{l}_{x}(1/k(t,y))dy / \int^{l}_{0}(1/k(t,y))dy

    Parameters:
    k (np.ndarray): Array containing the values of k(t,x) evaluated in the discretised spatial domain.
    x_eval (np.ndarray): Points used to discretise the domain.

    Returns:
    p (np.ndarry): Array containing the values of p(t,x) evaluated in the discretised spatial domain.
    """
    
    u = get_u(k,x_eval)
    p = np.zeros(len(x_eval))
    p[0] = 1
    for i in range(1,len(x_eval)-1):
        p[i] = integrate_inverse_k(k=k,x_eval=x_eval,i=i) * u
    return p

def get_psi(k,j,x_eval):
    """
    Find \psi(t,x) = j(t,x)/k(t,x) * u 

    Parameters:
    k (np.ndarray): Array containing the values of k(t,x) evaluated in the discretised spatial domain.
    x_eval (np.ndarray): Points used to discretise the domain.

    Returns:
    p (np.ndarry): Array containing the values of p(t,x) evaluated in the discretised spatial domain.
    """
    u = get_u(k,x_eval)
    k_inv = inverse_k(k)
    psi = j * k_inv * u
    return psi