import numpy as np
from solver import solver

t_eval = np.linspace(0,5,6)
x_sol, z_sol = solver(t_eval)
print(f'The solution for x at t=0 is {x_sol[0,:,:,:]}')
print(f'The solution for z at t=0 is {z_sol[0,:,:,:]}')