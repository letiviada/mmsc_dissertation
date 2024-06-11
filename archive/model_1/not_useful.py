from casadi import * 
import numpy as np
def cut_ode_array(arr,nx):
    """
    Cuts the 1D array into segments of sizes nx, nx+shape

    Parameters:
    arr (np.ndarray): The input 1D array of size (nx + nx*shape,).
    nx (int): The size of the spatial domain.

    Returns:
    list: A list containing the two segments as numpy arrays.
    """
    c = arr[:nx]
    G = arr[nx:]
    return c,G


def cut_algebraic_array(arr, num_i, nx):
    """
    Cuts the 1D array into segments of sizes num_i, nx, nx, nx, ..., 1.

    Parameters:
    arr (np.ndarray): The input 1D array of size (num_i + 4nx + 1,).
    num_i (int): The size of the first segment.
    nx (int): The size of the subsequent segments.

    Returns:
    list: A list containing the segments as numpy arrays.
    """
    segments = []
    start = 0
    
    # First segment of size W
    segments.append(arr[start:start+num_i])
    start += num_i
    
    # Subsequent segments of size n
    for _ in range(4):
        segments.append(arr[start:start+nx])
        start += nx
    
    # Last segment of size 1
    segments.append(arr[start:start+1])
    
    return segments

import casadi as ca
xgrid = np.linspace(1,6,6)
V = [-1,-1,-2,-3,0,2]
lut = ca.interpolant('LUT','bspline',[xgrid],V)
print(lut(2.5))
x = ca.MX.sym('x',(3,1))
print(lut(x))
