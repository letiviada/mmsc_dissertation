import casadi as ca
import numpy as np
from dae_system import ode_rhs, algebraic_equation

def solver_casadi(nx,dx):
    """Solves the DAE system using CasADi"""
    h = ca.SX.sym('h',nx)
    u = ca.SX.sym('u',nx) # Variable in differential equation
    rhs= ode_rhs(u,h,dx) # Define differential equation
    return rhs