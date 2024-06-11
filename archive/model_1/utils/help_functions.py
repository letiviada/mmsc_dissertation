import numpy as np 
from casadi import *
from functools import reduce

def inverse_k(k):
    """
    Calculate the inverse of k.

    Parameters:
    k (np.ndarray): Array of k values.
    
    Returns:
    k_inv (np.ndarray): Array of inverse k values
    """
    k_inv = 0.0 * SX.ones(k.shape)
    for i in range(k.shape[0]):
        k_inv[i] = 1/k[i]
    return k_inv
def integrate_k_simpson(k):
    """
    Integrate the inverse of k over the spatial domain using Simpson's Rule

    Parameters:
    k (np.ndarray): Array of k values.

    Returns: 

    integral_simpson (float): Result of integral
    """
    N = k.shape[0]
    if N == 1:
        return 0
    else:
        dx = 1 / (N - 1)
        integral_simpson = dx/3 * (k[0] + 4*sum1(k[1:N-1:2]) + 2*sum1(k[2:N-2:2]) + k[-1])
        return integral_simpson
def integrate_inverse_k(k,x_eval,i):
    """
    Integrates the inverse of the function k(t,x)

    Parameters:
    k (np.ndarray): Array of size (nx,1) with the entries at each point.
    x_eval (np.ndarray): Points used to discretise the domain
    i (int): The index in x_eval from which to start the integration.

    Returns:
    integ (float): Result of integrating k between x_start and l = 1
    """
    k_inv = inverse_k(k)
    # Ensure the index is within the valid range
    if i < 0 or i >= len(x_eval):
        raise ValueError("Index i is out of the range of x_eval")
    # Slice the arrays from index i to the end
    x_eval_slice = x_eval[i:]
    k_slice = k_inv[i:]

    # Integrate using the specified method
    result = integrate_k_simpson(k_slice) # DO CASADI FUNCTION
        #result = simpson(y=f_slice,x=x_eval_slice)

    return result

def sum_over_last_index(tensor_flat, original_shape):
    num_elements_per_subtensor = original_shape[-1]
    num_subtensors = int(shape_to_length(original_shape)/ num_elements_per_subtensor)
    # Initialize the result array
    result = SX.zeros(num_subtensors)
    
    # Sum over the last index for each subtensor
    for index in range(num_subtensors):
        start_idx = index * num_elements_per_subtensor
        end_idx = start_idx + num_elements_per_subtensor
        result[index] = sum1(tensor_flat[start_idx:end_idx])
    
    return result

def shape_to_length(shape):
    """
    Set up the CasADi integrator for solving the DAE system.
    
    Parameters:
    shape (tuple): shape of the cell (i,j,r,s)
    
    Returns:
    size (int): returns the lenth a 1d array has to have: i*j*r*s
    """
    size = reduce (lambda x,y: x*y,shape)
    return size

