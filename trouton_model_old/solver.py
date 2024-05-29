import casadi as ca
import numpy as np
from dae_system import ode_rhs, algebraic_equation
from bdry_initial_conditions import initial_u, boundary_conditions


def solver_casadi(nx,dx,BCS,t_span):
    """Solves the DAE system using CasADi"""
    h = ca.SX.sym('h',nx)
    u  = ca.SX.sym('u',nx)
    t = ca.SX.sym('t')
    u = boundary_conditions(u,BCS)
    #print(u[1:-1].shape)
    #u  = ca.SX.sym('u',nx)
    rhs= ode_rhs(u,h,dx) # Define differential equation
    #print(f'ode has size {rhs.shape}')
    alg = algebraic_equation(u,h,dx)
    #print(f'algebraic equations are {alg.shape} and are {alg}')
    dae = {'x':h,'z':u[1:-1],'t':t, 'ode': rhs, 'alg': alg}
    t_vals = np.linspace(t_span[0],t_span[1],101)
    integrator = ca.integrator('integrator', 'idas', dae, 0,t_vals)

    h0 = np.ones(nx)
    u0 = initial_u(nx,BCS)
    #print(u0)
    result = integrator(x0=h0, z0=u0)
    return result

def solve_initial(nx,dx,BCS,t_span):
    h = ca.SX.sym('h')
    u  = ca.SX.sym('u',nx)
    u[0] = 1.0
    u[-1] = 10.0
    alg_eqn = ca.SX(nx,1)
    alg_eqn[0] = 0.0
    alg_eqn[-1] = 0.0
    for i in range(1, nx-1):
        alg_eqn[i] = (u[i+1] - 2 * u[i] + u[i-1]) / (dx **2)
    
    dae = {'x':h,'z':u[1:-1], 'ode': 0, 'alg': alg_eqn}
    t_vals = np.linspace(t_span[0],t_span[1],10)
    integrator = ca.integrator('integrator', 'idas', dae, 0)
    h0 = np.ones(nx)
    u0 = initial_u(nx,BCS)

    result = integrator(x0=h0, z0=u0)
