import numpy as np
from solver import solver_casadi

# Number of spatial points
nx = 5
x = np.linspace(0, 1, nx)
dx = x[1] - x[0]

# Time span
t_span = [0, 10]

# Boundary conditions for u
BCS = [1, 10]

# Solve the system
result = solver_casadi(nx, dx, BCS, t_span)

# Extract results
h_res = result['xf'].full()
u_res = result['zf'].full()

# Print results
print("h:", h_res)
print("u:", u_res)
