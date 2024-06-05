from casadi import *
from model import ode, alg, initial

def solver(t_eval):
    nt = len(t_eval)
    nx = 144
    # Variables
    x = SX.sym('x',(nx,1))
    z = SX.sym('z',(nx,1))
    # Define the solver
    opts = {'reltol':1e-10,'abstol':1e-10}
    dae = {'x':x,'z':z, 'ode': ode(x,z), 'alg': alg(x,z)}
    F = integrator('F', 'idas', dae, t_eval[0],t_eval,opts)
    # Initial confitions given as a numpy array
    _, x0 = initial()
    z0 = 0.0*x0

    result = F(x0=x0, z0=z0)
    x_res = result['xf'].full()
    z_res = result['zf'].full()
    # Reshape the solutions to obtain tensors
    X = x_res.transpose().reshape(nt,9,4,4)
    Z = z_res.transpose().reshape(nt,9,4,4)
    return X, Z
