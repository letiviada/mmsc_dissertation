import casadi as ca
def ode_rhs(u,h,dx):
    """
    Defines the right-hand side of the ODE.
    Uses Method of Lines to discretize the spatial derivatives
    """
    uh = u*h
    #print(f'shape of u is {u.shape} and it is {u}')
    #print(f'shape of h is {h.shape} and it is {h}')
    #print(f'shape of uh is {uh.shape} and it is {uh}')
    # We use central differences for interior points
    uh_central = (uh[2:] - uh[:-2]) / (2 * dx)
    #print(f'shape of uh_central is {uh_central.shape} and it is {uh_central}')
    # We use backward finite differences for the right-most point
    uh_backward = (uh[-1] - uh[-2]) / dx
    #print(f'shape of uh_backward is {uh_backward.shape} and it is {uh_backward}')
    # We use forward finite differences for the left-most point
    uh_forward = (uh[1] - uh[0]) / dx
   # print(f'shape of uh_forward is {uh_forward.shape} and it is {uh_forward}')
    uh_x = ca.vertcat(uh_forward,uh_central, uh_backward)
    #print(f'shape of uh_x is {uh_x.shape} and it is {uh_x}')
    # We concatenate the finite differences
    return -uh_x
    

def algebraic_equation(u,h,dx):
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
   # print(f'shape of uh_forward is {uh_forward.shape} and it is {uh_forward}')
    hux = ca.vertcat(hu_x_forward,hu_x_central, hu_x_backward)
    #print(f'shape of hux is {hux.shape} and it is {hux}')
    return hux
