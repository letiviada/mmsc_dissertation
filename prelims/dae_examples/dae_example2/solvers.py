import casadi as ca
import numpy as np
from dae_system import dae_rhs, algebraic_equation

def solver_casadi(x0,t_span):
    """Solves the DAE system using CasADi"""
    x = ca.MX.sym('x') # Variable in differential equation
    y = ca.MX.sym('y')
    z = ca.MX.sym('z') # Variable in algebraic equation
    t = ca.MX.sym('t')
    x_all = ca.vertcat(x,y)
    diff = dae_rhs(t,x,y,z) # Define differential equation
    print(diff)
    print(diff.shape)
    alg_eq = algebraic_equation(x,z) #Define algebraic equation
    # DAE system
    dae = {'x': x_all, 'z': z, 't':t,'ode': ca.vertcat(diff[0],diff[1]), 'alg': alg_eq}
    # Integrator options
    opts = {
        'output_t0': True,
        'grid': np.linspace(t_span[0], t_span[1], 100) # Time grid for output
    }
    
    # Create integrator
    #integrator = ca.integrator('integrator', 'idas', dae, opts)
    integrator = ca.integrator('integrator', 'idas', dae, 0,np.linspace(t_span[0],t_span[1],100))
    # Initial conditions
    z0 = 2 * x0[0]
    # Simulate the system
    result = integrator(x0=x0, z0=[z0])
    t_vals = opts['grid']
    x_vals = result['xf'].full()#.flatten()
    z_vals = result['zf'].full().flatten()
    return t_vals, x_vals,z_vals