from casadi import *
import numpy as np
from utils.get_functions import get_k, get_u, get_p, get_psi

def alg_k(G,Delta,k):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        G: differentiable variable representing pore conductance
        Delta: algebraic variable representing pore pressure drop
        H: heaviside function 
        r: vector indicating the index
        k: algebraic variable for permeability
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_k: Algebraic equation used to define k
    """
    alg_eqn_k = SX(k.shape[0],1)
    alg_eqn_k = k - get_k(G=G, Delta=Delta,nx=k.shape[0])
    return alg_eqn_k


def alg_u(k,u,x_eval):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        x: k algebraic variable
        z: u algebraic variable
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_u: Algebraic equation used to define u
    """
    alg_eqn_u = SX(u.shape[0],1)
    alg_eqn_u = u - get_u(k=k,x_eval=x_eval)
    return alg_eqn_u

def alg_p(k,p,x_eval):
    """ Function that defines the algebraic equations of p.
    
    Parameters:
        k: Algebraic variable
        p: Algebraic variable
        x_eval (np.ndarray): Points in the spatial domain
     
    Returns:
        alg_eqn_p: Algebraic equation used to define p
    """

    alg_eqn_p = SX(p.shape[0],1)
    alg_eqn_p[:] = p[:] - get_p(k,x_eval)[:]

    return alg_eqn_p

def alg_psi(k,j,psi,x_eval):
    """ Function that defines the algebraic equations of u.
    
    Parameters:
        k: Algebraic variable (write physical meaning)
        j: Algebraic variable
        psi: Algebraic variable
        x_eval (np.ndarray): Points in the spatial domain

    Returns:
        alg_eqn_psi: Algebraic equation used to define psi
    """

    alg_eqn_psi = SX(psi.shape[0],1)
    alg_eqn_psi[:] = psi[:] - get_psi(k,j,x_eval)[:]

    return alg_eqn_psi

def ode(x,z):
    ode = SX(x.shape[0],1)
    return ode

x = SX.sym('x',(1,1))
u = SX.sym('u',(1,1))
k = SX.ones((3,1))
p = SX.sym('p',(3,1))
z = vertcat(u,p)
x_eval = np.linspace(0,1,3)
t_eval = np.linspace(0,5,6)

dxdt = ode(x,u)
alg_u = alg_u(k,u,x_eval)
alg_p = alg_p(k,p,x_eval)


alg = vertcat(alg_u,alg_p)


opts = {'reltol': 1e-10, 'abstol': 1e-10}
dae = {'x': x, 'z': z , 'ode':dxdt, 'alg': alg}
F = integrator('F', 'idas', dae, t_eval[0], t_eval, opts)

x0 = np.ones((1,1))
z0 = 0.0*np.ones((4,1))
result = F(x0=x0, z0=z0)
x_res = result['xf'].full()
z_res = result['zf'].full()
print(z_res)