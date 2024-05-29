import casadi as ca
import numpy as np

def alg(x,z,nx):
    """ Function that defines the algebraic equations of the DAE system of the Trouton Model.
    
    Args:
        x: Differentiable variable
        z: Algebraic variable
        nx: Number of points in the spatial domain
    """
    dx = 1 / (nx - 1)
    alg_eqn = ca.SX(nx,1)
    alg_eqn[0] = z[0] - 1.0
    alg_eqn[-1] = z[-1] - 10.0
    # Vectorized computation for the interior points
    term1 = (x[2:] - x[:-2]) * (z[2:] - z[:-2]) / (4 * dx**2)
    term2 = x[1:-1] * (z[2:] - 2 * z[1:-1] + z[:-2]) / (dx**2)
    alg_eqn[1:-1] = term1 + term2
    return alg_eqn

def ode(x,z,nx):
    """ Function that defines the differential equations of the DAE system of the Trouton Model.
    
    Args:
        x: Differentiable variable
        z: Algebraic variable
        nx: Number of points in the spatial domain
    """
    dx = 1 / (nx - 1)
    xz = x * z  #Multiplies element-wise

    dhdt=ca.SX(nx,1)
    #dhdt[0] = -(xz[1] - xz[0]) / dx
    #dhdt[0] = 0.0
    #dhdt[-1] = -(xz[-1] - xz[-2]) / dx
    dhdt[1:] = -(xz[1:] - xz[:-1]) / dx
    return dhdt

def initial(x,z,nx):
    x0 = np.ones((nx,1))
    z0 = 0.0 * np.ones((nx,1))
    return [x0,z0]
