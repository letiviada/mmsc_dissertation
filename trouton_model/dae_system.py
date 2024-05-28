import casadi as ca
import numpy as np
def ode_rhs1(u,h,dx):
    """
    Defines the right-hand side of the ODE.
    Uses Method of Lines to discretize the spatial derivatives
    """
    uh = u*h
    # We use central differences for interior points
    uh_central = (uh[2:] - uh[:-2]) / (2 * dx)
    # We use backward finite differences for the right-most point
    uh_backward = (uh[-1] - uh[-2]) / dx
    # We use forward finite differences for the left-most point
    uh_forward = (uh[1] - uh[0]) / dx
    uh_x = ca.vertcat(uh_forward,uh_central, uh_backward)
    return -uh_x

def ode_rhs(u, h, dx):
    """ Defines the right-hand side of the ODE for h """
    nx = h.shape[0]
    rhs = ca.SX.zeros(nx)
    for i in range(1, nx-1):
        rhs[i] = - (h[i] * u[i] - h[i-1] * u[i-1]) / dx
    return rhs
    
def algebraic_equations(u,h,dx):
    """Defines the algebraic equation."""
    # Define u_{x} using central differences for interior points
    u_central = (u[2:] - u[:-2]) / (2 * dx)
    #print(f'shape of u_central is {u_central.shape} and it is {u_central}')
    # Use Backward differences for the right-most point
    u_backward = (u[-1] - u[-2]) / dx
    #print(f'shape of u_backward is {u_backward.shape} and it is {u_backward}')
    # Use Backward differences for the right-most point
    u_forward = (u[1] - u[0]) / dx
    #print(f'shape of u_forward is {u_forward.shape} and it is {u_forward}')
    u_x = ca.vertcat(u_forward,u_central, u_backward)
    #print(f'shape of u_x is {u_x.shape} and it is {u_x}')
    hu_x = h*u_x

    #print(f'shape of hu_x is {hu_x.shape} and it is {hu_x}')
    # We use central differences for interior points
    hu_x_central = (hu_x[2:] - hu_x[:-2]) / (2 * dx)
    #print(f'shape of uh_central is {uh_central.shape} and it is {uh_central}')
    # We use backward finite differences for the right-most point
    hu_x_backward = (hu_x[-1] - hu_x[-2]) / dx
    #print(f'shape of uh_backward is {uh_backward.shape} and it is {uh_backward}')
    # We use forward finite differences for the left-most point
    hu_x_forward = (hu_x[1] - hu_x[0]) / dx
    #print(f'shape of uh_forward is {uh_forward.shape} and it is {uh_forward}')
    hux = ca.vertcat(hu_x_forward,hu_x_central, hu_x_backward)
    #print(f'shape of hux is {hux.shape} and it is {hux}')

    hux_central = (hux[2:]-hux[:-2])/(2 * dx)
    #print(f'shape of hux_central is {hux_central.shape} and it is {hux_central}')
    hux_forward = (hux[1]-hux[0])/ dx
    #print(f'shape of hux_forward is {hux_forward.shape} and it is {hux_forward}')
    hux_backward= (hux[-1]-hux[-2])/ dx
    #print(f'shape of hux_backward is {hux_backward.shape} and it is {hux_backward}')
    hux_x = ca.vertcat(hux_forward,hux_central,hux_backward)
    #print(f'shape of hux_x is {hux_x.shape} and it is {hux_x}')

    return hux_central

def algebraic_equation(u, h, dx):
    """ Defines the algebraic equations for u """
    nx = u.shape[0]
    alg_eqs = []
    for i in range(1, nx-1):
        term1 = (h[i+1] - h[i-1]) * (u[i+1] - u[i-1]) / (4 * dx * dx)
        term2 = h[i] * (u[i+1] - 2 * u[i] + u[i-1]) / (dx * dx)
        alg_eqs.append(term1 + term2)
    return ca.vertcat(*alg_eqs)
