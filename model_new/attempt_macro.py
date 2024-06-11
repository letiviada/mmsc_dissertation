from casadi import *
from utils.get_functions import integrate_simpson,reshape_f

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
# Spatial domain
nx = 15
x_eval = np.linspace(0,1,nx)

# Differential Variables
c = MX.sym('c', (nx,1))
tau = MX.sym('tau',(nx,1))

# Algebraic variables
u = MX.sym('u',(1,1))
psi = MX.sym('psi',(nx,1))

# Retrieve data from the microscale system
#k_fun = lambda x: x**2 +1
#k = k_fun(np.linspace(-10,10,11))
k  = np.ones(11)
k_inv = np.where(k != 0, 1/k, 0) # Obtain inverse of k
j = 2*np.ones(11)
tau_eval = np.linspace(-10,10,11)

# Interpolate
interp_k = interpolant('INTERP_K','linear',[tau_eval],k)
interp_k_inv = interpolant('INTERP_K_INV','linear',[tau_eval],k_inv)
interp_j = interpolant('INTERP_J','linear',[tau_eval],j)


# Define the system of differential equations
x = vertcat(c,tau)
ode = vertcat(ode_c(c,u,psi,nx),ode_tau(tau,c,u,interp_k))

# Define the system of algebraic equations
z = vertcat(u,psi)
alg = vertcat(alg_u(u,interp_k,tau),alg_psi(psi,u,interp_k_inv,interp_j,tau))

# Time domain
nt=11
t_eval = np.linspace(0,10,nt)

# Define solver
opts = {'reltol': 1e-8, 'abstol': 1e-8}
dae = {'x': x, 'z': z , 'ode':ode, 'alg': alg}
F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts) 

# Initial Conditions
c00 = 1.0
c0 = np.zeros((nx-1,1))
tau0 = np.zeros((nx,1))
x0 = vertcat(c00,c0,tau0)
z0 = np.zeros((nx+1,1))

# Solve problem 
result = F(x0=x0, z0=z0)
x_res = result['xf'].full()
z_res = result['zf'].full()


c,tau,u,psi = reshape_f(x_res,z_res,nt,nx)
print(c,tau)
print(u,psi)
