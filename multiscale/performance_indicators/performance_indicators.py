import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

class FilterPerformance:
    def __init__(self, t_eval: np.ndarray, u: np.ndarray,c: np.ndarray):
        self.t_eval = t_eval
        self.u = u
        self.c = c

    def throughput(self, tf: np.ndarray) -> np.ndarray:
        """
        Function to calculate the throughput of the filter given the darcy_velocity.
        This function also gives the lifetime of the filter if tf is the termination time.

        Parameters:
        -----------
        t_eval (np.ndarray): Time array of the filter simulation.
        tf (np.ndarray): Time of interest of the thorugput
        u (np.ndarray): Darcy velocity of the filter.

        Returns:
        --------
        throughput (np.ndarray): The throughput of the filter.
        """
        velocity = interp1d(self.t_eval, self.u, kind='cubic', fill_value='extrapolate')
        throughput = np.array([quad(velocity, 0, time, limit = 75)[0] for time in tf])
        return throughput

    def efficiency(self,t_eval) -> np.ndarray:
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
        c_outlet= self.c[:,-1]
        concent_outlet_interp = interp1d(self.t_eval,c_outlet,kind='cubic',fill_value='extrapolate')
        conc_outlet = concent_outlet_interp(t_eval)
        removed_particles =  1 - conc_outlet

        avg_efficiency = (quad(concent_outlet_interp, 0, t_eval[-1])[0]) / (t_eval[-1])
        avg_removed_particles = 1 - avg_efficiency
        return conc_outlet,removed_particles, avg_efficiency, avg_removed_particles

    def termination_time(self, mu: float) -> float:
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
        velocity = interp1d(self.t_eval, self.u, kind='cubic', fill_value='extrapolate')
        def velocity_minus_mu(t):
            return velocity(t) - mu
        if velocity_minus_mu(0) * velocity_minus_mu(self.t_eval[-1])>0:
            return self.t_eval[-1] 
        else:
            termination_time = root_scalar(velocity_minus_mu, bracket=[0, self.t_eval[-1]])
            if termination_time.converged:
                return termination_time.root
            else:
                return None
