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
    term1 = (x[2:] + x[1:-1]) * (z[2:] - z[1:-1]) / (2 * dx ** 2)
    term2 = (x[1:-1] + x[:-2]) * (z[1:-1] - z[:-2]) / (2 * dx ** 2)
    alg_eqn[1:-1] = term1 - term2
    return alg_eqn


def ode(x,z,nx):
    """ Function that defines the differential equations of the DAE system of the Trouton Model.
    We use an upwind scheme for the parabolic equation, so that the information propagates from the previous space point to the following
    Then, (uh)_{x}(x_{j},t) = ((UH)_{j}-(UH)_{j-1})/(dx).

    Args:
        x: Differentiable variable
        z: Algebraic variable
        nx: Number of points in the spatial domain
    """
    dx = 1 / (nx - 1)
    xz = x * z  #Multiplies element-wise

    dhdt=ca.SX(nx,1)
    dhdt[1:] = -(xz[1:] - xz[:-1]) / dx #Backward differences
    return dhdt

def initial(x,z,nx):
    x0 = np.ones((nx,1))
    z0 = 0.0 * np.ones((nx,1))
    return [x0,z0]
