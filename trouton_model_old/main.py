import numpy as np
from solver import solver_casadi
import matplotlib.pyplot as plt

# Number of spatial points
nx = 100
x = np.linspace(0, 1, nx)
dx = x[1] - x[0]

# Time span
t_span = [0, 5]

# Boundary conditions for u
BCS = [1, 10]

# Solve the system
result = solver_casadi(nx, dx, BCS, t_span)

# Extract results
h_res = result['xf'].full()
u_res = result['zf'].full()
h_array = np.array(h_res)
#print(u_res)
#print(h_res)
time_points = np.linspace(t_span[0],t_span[1],h_res.shape[1])
plt.figure()
for i in range(h_res.shape[0]):
    plt.plot(time_points, h_res[i, :], label=f'h at x{i}')

#for i in range(h_res.shape[1]):
 #   plt.plot(x, h_array[:, i], label=f'h at x{i}')
#plt.xlabel('Time')
plt.xlabel('Space x')
plt.ylabel('h')
#plt.title('h as a function of time')
plt.title('h as a function of space')
#plt.legend()
plt.show()

