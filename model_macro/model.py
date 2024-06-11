from casadi import * 
from utils.get_functions import integrate_simpson

def ode_c(c,u,psi,nx,phi=1):
    """
    Differential equation 

    Parameters:
    c (MX): Differentiable variable 
    u (MX): Algebraic variable representing Darcy's velocity
    psi (MX): Algebraic variable
    nx (int): Number of spatial points
    phi(float):

    Returns:
    dcdt (MX): RHS after upwinding in space
    """

    dx = 1 / (nx - 1) # Spatial domain size
    dcdt = MX(nx,1) # Define vector or RHS
    dcdt[1:] = -(u / phi) * ((c[1:]-c[:-1])/dx) - (psi[1:] / phi) * c[1:] # Use upwinding scheme to define each RHS
    #dcdt[1:] = 0.5
    return dcdt

def ode_tau(tau,c,u,interp_k,nx):
    """
    Differential equation for the auxiliar variable tau

    Parameters:
    tau (MX): Differentiable variable for the auxiliar variable
    c (MX): Differentiable variable
    u (MX): Algebraic variable representing Darcy's velocity
    k (np.ndarray): Array from the microscale problem
    tau_eval (np.ndarray): tau points k is evaluated in

    Returns:
    dtaudt (MX): RHS of equation

    """
    dtaudt = MX(nx,1)
    k_MX = interp_k(tau) #Obtain interpolated MX type of k
    #dtaudt[:] = 1.0
    dtaudt[:] = c[:] * u * k_MX[:] # Define the RHS of the ODE
    return dtaudt

# ----------------------------- # ----------------------------#

def alg_u(u,interp_k_inv,tau):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        u (MX): Algebraic variable for Darcy's velocity
        k (np.ndarrray): u algebraic variable
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_u: Algebraic equation used to define u
    """
    k_inv_MX = interp_k_inv(tau)  # Obtain k as np.ndarray and shape (nx,1)
    u_inv = integrate_simpson(k_inv_MX)
    u_fl = 1 / u_inv # Define u
    u_MX = MX(u_fl) # Maki it SX type

    alg_eqn_u = MX(u.shape[0],1)
    alg_eqn_u = u - u_MX # Define algebraic equation
    return alg_eqn_u

def alg_psi(psi,u,interp_k_inv,interp_j,tau):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        k: Algebraic variable (write physical meaning)
        j: Algebraic variable
        psi: Algebraic variable
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_psi: Algebraic equation used to define psi
    """
    k_MX_inv = interp_k_inv(tau)
    j_MX = interp_j(tau) # Obtain MX type for j in shape (nx,1)
  
    psi_alg = j_MX * k_MX_inv * u # Obtain equation psi satisfies

    alg_eqn_psi = MX(psi.shape[0],1)
    alg_eqn_psi[:] = psi[:] - psi_alg[:] # Define the algebraic equation at each spatial point

    return alg_eqn_psi

# ----------------------------- # ----------------------------#