import numpy as np
def dae_rhs(t,x_array,z):
    """Defines the right-hand side of the DAE."""
    dxdt = 0.5
    dydt = x_array[1]
    dmdt = x_array[2]
    return np.array([dxdt,dydt,dmdt])

def algebraic_equation(x,z):
    """Defines the algebraic equation."""
    f_alg = z - 2 * x[0] # Make it of the form f_alg == 0
    return f_alg