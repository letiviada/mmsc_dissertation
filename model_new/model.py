from casadi import *
from utils.get_functions import interp1, reshape_ode_array, reshape_alg_array, integrate_simpson

def ode_c(c,u,psi,nx,phi=1):
    """
    Differential equation 

    Parameters:
    c (MX): Differentiable variable 
    u (MX): Algebraic variable representing Darcy's velocity

    """

    dx = 1 / (nx - 1) # Spatial domain size
    dcdt = MX(nx,1) # Define vector or RHS
    dcdt[1:] = -(u / phi) * ((c[1:]-c[:-1])/dx) - (psi[1:] / phi) * c[1:] # Use upwinding scheme to define each RHS
    return dcdt

def ode_tau(tau,c,u,k,tau_eval,x_eval):
    """

    """
    dtaudt = MX(nx,1)
    k_SX = interp1(k,tau_eval,tau) #Obtain interpolated MX type of k
    dtaudt[:] = c[:] * u * k_SX[:] # Define the RHS of the ODE
    return dtaudt

#######################################################

def alg_u(u,k,tau_eval,x_eval):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        x: k algebraic variable
        z: u algebraic variable
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_u: Algebraic equation used to define u
    """
    k_inv = np.where(k != 0, 1/k, 0) # Obtain inverse of k
    k_inv_MX = interp1(k_inv,tau_eval,tau) # Obtain k as np.ndarray and shape (nx,1)
   #k_inv  = np.ones((len(x_eval)))
    #u_inv = simpson(y = k_inv,x = x_eval) # Integrate the inverse
    u_fl = integrate_simpson(k_inv_MX)
    #u_inv = MX.ones((u.shape[0],1))
    #u_fl = u_inv
    #u_fl = 1 / u_inv # Define u
    u_SX = MX(u_fl) # Maki it SX type

    alg_eqn_u = MX(u.shape[0],1)
    alg_eqn_u = u - u_SX # Define algebraic equation
    return alg_eqn_u

def alg_psi(psi,u,k,j,tau_eval,x_eval):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        k: Algebraic variable (write physical meaning)
        j: Algebraic variable
        psi: Algebraic variable
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_psi: Algebraic equation used to define psi
    """
    #_,k_numpy = interp(k,tau_eval,x_eval) # Obtain k as np.ndarray and shape (nx,1)
    k_inv = np.where(k!=0,1/k,0) # Obtain inverse of k
    k_MX_inv = interp1(k_inv,tau_eval,tau)
    #k_SX_inv = MX(k_inv) # Make it SX type
    #j_SX,_ = interp(j,tau_eval,x_eval) # Obtain SX type for j in shape (nx,1)
    j_MX = interp1(j,tau_eval,tau)
  
    psi_alg = j_MX * k_MX_inv * u # Obtain equation psi satisfies

    alg_eqn_psi = MX(psi.shape[0],1)
    alg_eqn_psi[:] = psi[:] - psi_alg[:] # Define the algebraic equation at each spatial point

    return alg_eqn_psi

# Spatial domain
nx = 5
x_eval = np.linspace(0,1,nx)

# Differential Variables
c = MX.sym('c', (nx,1))
tau = MX.sym('tau',(nx,1))

# Algebraic variables
u = MX.sym('u',(1,1))
psi = MX.sym('psi',(nx,1))

# Retrieve data from the microscale system
k = np.ones(101)
j = 2*np.ones(101)
tau_eval = np.linspace(-10,10,101)

# Define the system of differential equations
x = vertcat(c,tau)
ode = vertcat(ode_c(c,u,psi,nx),ode_tau(tau,c,u,k, tau_eval,x_eval))

# Define the system of algebraic equations
z = vertcat(u,psi)
alg = vertcat(alg_u(u,k,tau_eval,x_eval),alg_psi(psi,u,k,j,tau_eval,x_eval))

# Time domain
nt=11
t_eval = np.linspace(0,10,nt)

# Define solver
opts = {'reltol': 1e-10, 'abstol': 1e-10}
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

# Reshape solution

c,tau = reshape_ode_array(x_res,nt,nx)
u, psi = reshape_alg_array(z_res,nt,nx)
print(c,tau)
print(u,psi)