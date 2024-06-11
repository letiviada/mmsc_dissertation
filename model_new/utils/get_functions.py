import numpy as np
from casadi import *
def integrate_simpson(k, l=1):
    """
    Integrate the inverse of k over the spatial domain using Simpson's Rule

    Parameters:
    k (np.ndarray): Array of k values.

    Returns: 

    integral_simpson (float): Result of integral
    """
    N = k.shape[0]
    dx = l / (N - 1)
    integral_simpson = dx/3 * (k[0] + 4*sum1(k[1:N-1:2]) + 2*sum1(k[2:N-2:2]) + k[-1])
    return integral_simpson

def interp1(f,tau_eval, x_eval):
    """"
    Function that interpolates np.ndarray and returns SX and np.ndarray in required spatial domain

    Parameters:

    f (np.ndarray): Function evaluated in tau_eval points.
    tau_eval (np.ndarray): Discretised domain where f is found
    x_eval (np.ndarray): Domain of the solution

    Returns:

    f_SX: Interpolated function as CasADi SX type
    f_numpy: Interpolated function as np.ndarray type
    """

    interp = interpolant('INTERP','bspline',[tau_eval],f)
    f_SX = interp(x_eval)
    return f_SX
