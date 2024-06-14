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
def reshape_f(x_res,z_res,nt,nx):
    """
    Reshapes the solution to and cuts it to obtain c and tau

    Parameters:
    arr (np.ndarray): The input 2D array of size (nt,2*nx).
    nt (int): Size of time domain.
    nx (int): The size of the spatial domain.

    Returns:
    list: A list containing the two segments as numpy arrays.
    """
    arr_x = x_res.transpose().reshape((nt,2*nx))
    arr_z = z_res.transpose().reshape((nt,nx+1))
    c = arr_x[:,:nx]
    tau = arr_x[:,nx:]
    u = arr_z[:,0]
    psi = arr_z[:,1:]
    return c, tau, u, psi
