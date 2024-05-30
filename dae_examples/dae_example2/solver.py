import casadi as ca
import numpy as np
from dae_system import dae_rhs, algebraic_equation
from scipy.optimize import fsolve

def solver_casadi(x0,t_span,n_diff_eqn,initial_guess=0):
    """Solves the DAE system using CasADi"""
    x = ca.MX.sym('x',n_diff_eqn) # Variable in differential equation
    z = ca.MX.sym('z') # Variable in algebraic equation
    t = ca.MX.sym('t')
    diff = dae_rhs(t,x,z) # Define differential equation
    alg_eq = algebraic_equation(x,z) #Define algebraic equation
    # DAE system
    dae = {'x': x, 'z': z, 't':t,'ode': ca.vertcat(*diff), 'alg': alg_eq}
    t_vals =np.linspace(t_span[0],t_span[1],100)
    integrator = ca.integrator('integrator', 'idas', dae, 0,t_vals)
    F = ca.Function('F', [z, x], [alg_eq]) # Create a function for the equation
    rootfinder = ca.rootfinder('rootfinder', 'newton', F) # Create a root finder using Newton's method
    z0 = rootfinder(initial_guess, x0) # Solve the root-finding problem with an initial guess for z and known x
    # Simulate the system
    result = integrator(x0=x0, z0=z0)
    x_vals = result['xf'].full()
    z_vals = result['zf'].full().flatten()
    return t_vals, x_vals,z_vals