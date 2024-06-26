from casadi import *
import numpy as np
from macro_model import MacroscaleModel

class Solver:
    def __init__(self,l):
        self.model = MacroscaleModel(l)  

    def setup(self,interp_k,interp_k_inv,interp_j,t_eval,nx,l,phi):
        """
        Sets up the differential-algebraic equation (DAE) solver with
        the necessary parameters and returns the integrator function.

        Parameters:
        ----------
            interp_k: Interpolation function for a parameter k.
            interp_k_inv: Inverse interpolation function for the parameter k_inv.
            interp_j: Interpolation function for a parameter j.
            t_eval (np.ndarray): Time points at which the solution should be evaluated.
            nx (int): Number of spatial points.
            l (float): Length of the filter.
            

        Returns:
        -------
            F: The integrator function that solves the DAE.
        """
        
        # Differential Variables
        c = MX.sym('c', (nx,1))
        tau = MX.sym('tau',(nx,1))

        # Algebraic variables
        u = MX.sym('u',(1,1))
        psi = MX.sym('psi',(nx,1))

        # Define the system of differential equations
        # --------------------------------------------
        x = vertcat(c,tau)
        ode = vertcat(self.model.ode_c(c,u,psi,nx,phi),self.model.ode_tau(tau,c,u,interp_k_inv,nx))

        # Define the system of algebraic equations
        # -----------------------------------------
        z = vertcat(u,psi)
        alg = vertcat(self.model.alg_u(u,interp_k_inv,tau,l),self.model.alg_psi(psi,u,interp_k,interp_k_inv,interp_j,tau))

        # Define solver
        # -------------
        opts = {'reltol': 1e-6, 'abstol': 1e-6}
        dae = {'x': x, 'z': z , 'ode':ode, 'alg': alg}
        F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts) 
        return F
    
    def run(self,F,nx: int)-> tuple:
        """
        Solves the DAE using the integrator function F.

        Parameters:
        ----------
            F: The integrator function that solves the DAE.
            nx (int): Number of spatial points.
        
        Returns:
        -------
            x_res (np.ndarray): The solution of the differential variables.
            z_res (np.ndarray): The solution of the algebraic variables.
        """

        # Initial Conditions
        # ------------------

        c00 = 1.0
        c0 = np.zeros((nx-1,1))
        tau0 = np.zeros((nx,1))
        x0 = vertcat(c00,c0,tau0)
        z0 = np.zeros((nx+1,1))

        # Solve problem 
        # -------------
        result = F(x0=x0, z0=z0)
        x_res = result['xf'].full()
        z_res = result['zf'].full()
        return (x_res,z_res)
