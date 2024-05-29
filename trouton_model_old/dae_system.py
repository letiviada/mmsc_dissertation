import casadi as ca
import numpy as np

def ode_rhs1(u,h,dx):
    """
    Defines the right-hand side of the ODE.
    Uses Method of Lines to discretize the spatial derivatives
    """
    uh = u*h
    #print(uh)
    # We use central differences for interior points
    uh_central = (uh[2:] - uh[:-2]) / (2 * dx)
    # We use backward finite differences for the right-most point
    uh_backward = (uh[-1] - uh[-2]) / dx
    # We use forward finite differences for the left-most point
    uh_forward = (uh[1] - uh[0]) / dx
    uh_x = ca.vertcat(-uh_forward,-uh_central, -uh_backward)
    #print(f'uh_x is {uh_x}')
    return uh_x

def ode_rhs(u, h, dx):
    """ Defines the right-hand side of the ODE for h """
    nx = h.shape[0]
    rhs = ca.SX.zeros(nx)
    for i in range(1, nx-1):
        rhs[i] = - (h[i] * u[i] - h[i-1] * u[i-1]) / (dx)
    print(f'rhs is {rhs}')
    return rhs
    
def algebraic_equation(u, h, dx):
    """ Defines the algebraic equations for u """
    nx = u.shape[0]
    alg_eqs = []
    for i in range(1, nx-1):
        term1 = (h[i+1] - h[i-1]) * (u[i+1] - u[i-1]) / (4 * dx**2)
        term2 = h[i] * (u[i+1] - 2 * u[i] + u[i-1]) / (dx **2)
        alg_eqs.append(term1 + term2)
    alg_eqn = ca.vertcat(*alg_eqs)
    return alg_eqn
