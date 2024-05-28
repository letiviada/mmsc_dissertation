import casadi as ca
import numpy as np

def boundary_conditions(u,BCS):
    """ Implements the boundary conditions of the system"""
    u[0] = BCS[0]
    u[-1] = BCS[1]
    return u

def initial_u(N,BCS):
    """ Implements the initial conditions of the system"""
    # Initialize the matrix A and vector b
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # Set up the boundary conditions
    b[0] = BCS[0]
    b[-1] = BCS[1]
    
    # Fill the matrix A
    for i in range(1,N-1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
    A[0,0] = 1
    A[-1,-1]=1
    # Solve the linear system
    u = np.linalg.solve(A, b)
    u_interior = u[1:-1]
    
    return u_interior