import numpy as np
import casadi as ca
def ode_rhs(u,h,dx):
    """
    Defines the right-hand side of the ODE.
    Uses Method of Lines to discretize the spatial derivatives
    """
    uh = u*h
    print(f'shape of u is {u.shape} and it is {u}')
    print(f'shape of h is {h.shape} and it is {h}')
    print(f'shape of uh is {uh.shape} and it is {uh}')
    # We use central differences for interior points
    uh_central = (uh[2:] - uh[:-2]) / (2 * dx)
    print(f'shape of uh_central is {uh_central.shape} and it is {uh_central}')
    # We use backward finite differences for the right-most point
    uh_backward = (uh[-1] - uh[-2]) / dx
    print(f'shape of uh_backward is {uh_backward.shape} and it is {uh_backward}')
    # We use forward finite differences for the left-most point
    uh_forward = (uh[1] - uh[0]) / dx
    print(f'shape of uh_forward is {uh_forward.shape} and it is {uh_forward}')
    uh_x = ca.vertcat(uh_forward,uh_central, uh_backward)
    print(f'shape of uh_x is {uh_x.shape} and it is {uh_x}')
    # We concatenate the finite differences
    return -uh_x
    

def algebraic_equation(x,z):
    """Defines the algebraic equation."""
    f_alg = z - 2 * x[0] # Make it of the form f_alg == 0
    return f_alg