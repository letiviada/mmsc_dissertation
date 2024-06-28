import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


def throughput(t_eval:np.ndarray,tf: float,u:np.ndarray)->float:
    """
    Function to calculate the throughput of the filter given the darcy_velocity.
    This function also gives the lifetime of the filter if tf is the termination time.

    Parameters:
    -----------
    t_eval (np.ndarray): Time array of the filter simulation.
    tf (float): Time of interest of the thorugput
    u (np.ndarray): Darcy velocity of the filter.

    Returns:
    --------
    throughput (float): The throughput of the filter.
    """
    velocity = interp1d(t_eval,u, kind='cubic', fill_value='extrapolate')

    throughput, _ = quad(velocity, 0, tf)    
    return throughput

def efficiency(t_eval:np.ndarray,c:np.ndarray)->np.ndarray:
    """
    Function to calculate the efficiency of the filter given the darcy_velocity.

    Parameters:
    -----------
    t_eval (np.ndarray): Time array of the simulation.
    c (np.ndarray): Concentration of the filter
    Returns:
    --------
    efficiency (np.ndarray): The efficiency of the filter.
    """
    efficiency = np.empty(shape=(len(t_eval),1))
    efficiency[:] = 1 - c[:,-1]
    return efficiency

def termination_time(t_eval:np.ndarray, u:np.ndarray, mu: float)->float:
    """
    Function to calculate the termination time of the filter given the darcy_velocity.

    Parameters:
    -----------
    u (np.ndarray): Darcy velocity of the filter.
    mu (float): Minimum allowed velocity.

    Returns:
    --------
    tf (float): The termination time of the filter.
    """
    velocity = interp1d(t_eval,u, kind='cubic', fill_value='extrapolate')
    def velocity_minus_mu(t):
        return velocity(t) - mu
    termination_time = root_scalar(velocity_minus_mu, bracket=[0, t_eval[-1]])
    if termination_time.converged:
        return termination_time.root
    else:
        return None
