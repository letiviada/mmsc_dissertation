from casadi import * 
from utils.get_functions import integrate_simpson

class MacroscaleModel:
    def __init__(self,l:float):
        self.l = l

    def ode_c(self,c: MX,u: MX,psi,nx: int,phi: float=1) -> MX:
        """
        Differential equation for concentration c.

        Parameters:
        -----------
            c (MX): Differentiable variable 
            u (MX): Algebraic variable representing Darcy's velocity
            psi (MX): Algebraic variable
            nx (int): Number of spatial points
            phi(float): Porosity, default value is 1

        Returns:
        --------
            dcdt (MX): RHS after upwinding in space
        """

        dx = 1 / (nx - 1) # Spatial domain size
        dcdt = MX(nx,1) # Define vector or RHS
        dcdt[1:] = -(u / phi) * ((c[1:]-c[:-1])/dx) - (psi[1:] / phi) * c[1:] # Use upwinding scheme to define each RHS
        #dcdt[1:] = 0.5
        return dcdt

    # ----------------------------- # ----------------------------#

    def ode_tau(self,tau: MX,c: MX,u: MX ,interp_k,nx:int) -> MX:
        """
        Differential equation for the auxiliar variable tau

        Parameters:
        tau (MX): Differentiable variable for the auxiliar variable
        c (MX): Differentiable variable
        u (MX): Algebraic variable representing Darcy's velocity
        interp_k (function): Interpolated function for k
        nx (int): Number of spatial points

        Returns:
        dtaudt (MX): differential equation for tau.

        """
        dtaudt = MX(nx,1)
        k_MX = interp_k(tau) #Obtain interpolated MX type of k
        #dtaudt[:] = 1.0
        dtaudt[:] = c[:] * u * k_MX[:] # Define the RHS of the ODE
        return dtaudt

    # ----------------------------- # ----------------------------#

    def alg_u(self,u: MX,interp_k_inv,tau: MX,l:float) -> MX:
        """ Function that defines the algebraic equations of u.
        
        Parameters:
        ----------
            u (MX): Algebraic variable for Darcy's velocity
            interp_k_inv (function): Interpolated function for k_inv
            tau (MX): Differentiable variable tau
            l (float): Length of the filter

        Returns:
        --------
            alg_eqn_u (MX): Algebraic equation used to define u
        """
        k_inv_MX = interp_k_inv(tau)  # Obtain k as np.ndarray and shape (nx,1)
        u_inv = integrate_simpson(k_inv_MX,l)
        u_fl = 1 / u_inv # Define u
        u_MX = MX(u_fl) # Maki it SX type

        alg_eqn_u = MX(u.shape[0],1)
        alg_eqn_u = u - u_MX # Define algebraic equation
        return alg_eqn_u

    # ----------------------------- # ----------------------------#

    def alg_psi(self,psi: MX,u: MX,interp_k_inv,interp_j,tau:MX) -> MX:
        """ Function that defines the algebraic equations of u.
        
        Parameters:
        ----------
            psi (MX): Algebraic variable for reactivity
            u (MX): Algebraic variable for Darcy's velocity
            interp_k_inv (function): Interpolated function for k_inv
            interp_j (function): Interpolated function for j
            tau (MX): Differentiable variable tau


        Returns:
        --------
            alg_eqn_psi (MX): Algebraic equation used to define reactivity, psi
        """
        k_MX_inv = interp_k_inv(tau)
        j_MX = interp_j(tau) # Obtain MX type for j in shape (nx,1)
    
        psi_alg = j_MX * k_MX_inv * u # Obtain equation psi satisfies

        alg_eqn_psi = MX(psi.shape[0],1)
        alg_eqn_psi[:] = psi[:] - psi_alg[:] # Define the algebraic equation at each spatial point

        return alg_eqn_psi

    # ----------------------------- # ----------------------------#