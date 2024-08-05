import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

class Pressure:
    def __init__(self, t_eval: np.ndarray,x_eval: np.ndarray, k: np.ndarray, u: np.ndarray,):
        self.t_eval = t_eval
        self.x_eval = x_eval
        self.k = k
        self.u = u
    
    def pressure(self) -> np.ndarray:
        """
        Function to calculate the pressure of the filter given the permeability.

        Parameters:
        -----------
        tf : np.ndarray
            Array of size (nt, nx) representing the permeability.

        Returns:
        --------
        np.ndarray
            Array of size (nt, nx) representing the calculated pressure.
        """

        pressure = np.zeros((self.t_eval.shape[0], self.x_eval.shape[0]))  # Initialize pressure array
        k_inv = np.where(self.k!=0, 1/self.k, 0)
        for i in range(self.t_eval.shape[0]):
           k_inv_interp = interp1d(self.x_eval, k_inv[i,:], kind='cubic', fill_value='extrapolate')
           pressure_den = quad(k_inv_interp, 0, self.x_eval[-1])[0]
           pressure_x = np.zeros(self.x_eval.shape[0])
           for xi in range(self.x_eval.shape[0]):
               pressure_x[xi] = quad(k_inv_interp,  self.x_eval[xi], self.x_eval[-1],limit = 75)[0]           
           pressure[i,:] = pressure_x[:] / pressure_den
        return pressure
    def pressure_grad(self) -> np.ndarray:
        """
        Function that gets the pressure gradient.

        Returns:
        --------
        np.ndarray
            Array of size (nt, nx) representing the pressure gradient.
        """
        u_big = self.u[:, np.newaxis]
        u_big = np.repeat(u_big, self.x_eval.shape[0], axis=1)
        pressure_grad = np.divide(-u_big, self.k)
        return pressure_grad