from casadi import *
import numpy as np
from model import ode_c, ode_tau, alg_u, alg_psi

def solv(interp_k,interp_k_inv,interp_j,t_eval,nx):

    # Differential Variables
    c = MX.sym('c', (nx,1))
    tau = MX.sym('tau',(nx,1))

    # Algebraic variables
    u = MX.sym('u',(1,1))
    psi = MX.sym('psi',(nx,1))

    # Define the system of differential equations
    x = vertcat(c,tau)
    ode = vertcat(ode_c(c,u,psi,nx),ode_tau(tau,c,u,interp_k,nx))

    # Define the system of algebraic equations
    z = vertcat(u,psi)
    alg = vertcat(alg_u(u,interp_k,tau),alg_psi(psi,u,interp_k_inv,interp_j,tau))

    # Define solver
    opts = {'reltol': 1e-10, 'abstol': 1e-10}
    dae = {'x': x, 'z': z , 'ode':ode, 'alg': alg}
    F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts) 
    return F
   
def run(F,nx):
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
    return x_res,z_res