from casadi import * 
from model import  ode_c, ode_tau, alg_u, alg_psi
     
def solver_dae(k,j,tau_eval,x_eval, t_eval):
    """
    Set up the CasADi integrator for solving the DAE system.
    
    Parameters:
    x_eval (np.ndarray): Spatial points.
    t_eval (np.ndarray): Time evaluation points.
    
    Returns:
    Function: CasADi integrator function.
    """
    # Number of points
    nx = len(x_eval)
    # Differential Variables
    c = MX.sym('c', (nx,1))
    tau = MX.sym('tau',(nx,1))

    # Algebraic variables
    u = MX.sym('u',(1,1))
    psi = MX.sym('psi',(nx,1))

    # Define the system of differential equations
    x = vertcat(c,tau)
    ode = vertcat(ode_c(c,u,psi,nx),ode_tau(tau,c,u,k,tau_eval,nx))

    # Define the system of algebraic equations
    z = vertcat(u,psi)
    alg = vertcat(alg_u(u,k,tau,tau_eval),alg_psi(psi,u,k,j,tau,tau_eval))
    
    opts = {'reltol': 1e-10, 'abstol': 1e-10}
    dae = {'x': x, 'z': z, 'ode': ode, 'alg': alg}
    F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts)
    return F

def run(F,x0,z0):
    """
    Run the CasADi integrator given initial conditions.

    Parameters:
    F (Function): CasADi integrator function.
    x0 (np.ndarray): Initial conditions for state variables.
    z0 (np.ndarray): Initial conditions for algebraic variables.
    """
    result = F(x0=x0, z0=z0)
    x_res = result['xf'].full()
    z_res = result['zf'].full()

    return x_res, z_res

def reshape(x_res,z_res,nt,nx):
    """
    Reshapes the solution to and cuts it to obtain c and tau

    Parameters:
    x_res (np.ndarray): The input 2D array of size (nt,2*nx).
    z_res (np.ndarray): The input 2D array of size (nt,nx+1)
    nt (int): Size of time domain.
    nx (int): The size of the spatial domain.

    Returns:
    list: A list containing the two segments as numpy arrays.
    """
    arr_x = x_res.transpose().reshape((nt,2*nx))
    c = arr_x[:,:nx]
    tau = arr_x[:,nx:]
    arr_z = z_res.transpose().reshape((nt,nx+1))
    u = arr_z[:,0]
    psi = arr_z[:,1:]
    return c,tau,u, psi
  