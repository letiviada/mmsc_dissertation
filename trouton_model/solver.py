import casadi as ca
import numpy as np
from model import ode, alg, initial
from utils.reshape_data import reshape

def solver(nx,t_eval):
    x = ca.SX.sym('x',(nx,1))
    z  = ca.SX.sym('z',(nx,1))
    # Define the solver
    opts = {'reltol':1e-6,'abstol':1e-6}
    dae = {'x':x,'z':z, 'ode': ode(x,z,nx), 'alg': alg(x,z,nx)}
    F = ca.integrator('F', 'idas', dae, t_eval[0], t_eval, opts)
    # Add initial conditions
    x0 = initial(x,z,nx)[0]
    z0 = initial(x,z,nx)[1]
    result = F(x0=x0, z0=z0)
    x_res = result['xf'].full()
    z_res = result['zf'].full()
    x_sol = reshape(x_res)
    z_sol = reshape(z_res)
    return x_sol, z_sol