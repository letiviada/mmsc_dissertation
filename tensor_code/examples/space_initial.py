import numpy as np
from casadi import *
import matplotlib.pyplot as plt

array_length = 144
nx = 11
nt = 6
def alg(x,z):
    alg_eqn = z-0.5*x
    return alg_eqn

def ode(x,z):
    ode = - x * z
    return ode

def initial(nx=1,t=0):
    tensor = np.zeros((nx,9, 4, 4)) # Shape of the tensor (r,i,j)
    def func(val,xi=0):
        c = 2 / val
        return 2 / (t + xi + c)
    values_dict = {
    (1, 0, 2): 1, (1, 1, 3): 2,
    (3, 0, 1): 3, (3, 2, 3): 4,
    (4, 0, 1): 5, (4, 0, 2): 6, (4, 1, 0): 5, (4, 1, 3): 7, (4, 2, 0): 6, (4, 2, 3): 8, (4, 3, 1): 7, (4, 3, 2): 8,
    (5, 1, 0): 9, (5, 3, 2): 10,
    (7, 2, 0): 11, (7, 3, 1): 12
    }
    # Assign non-zero values to the tensor
    x_axis = np.linspace(0,1,nx)
    for i, xi in enumerate(x_axis):
        for pos, val in values_dict.items():
            tensor[i, pos[0], pos[1], pos[2]] = func(val, xi =xi)

    in_cond = tensor.reshape(-1)
    return tensor, in_cond


x = SX.sym('x',(array_length*nx,1))
z = SX.sym('z',(array_length*nx,1))
t_eval = np.linspace(0,5,nt)
x_eval = np.linspace(0,1,nx)

opts = {'reltol':1e-10,'abstol':1e-10}
dae = {'x':x,'z':z, 'ode': ode(x,z), 'alg': alg(x,z)}
F = integrator('F', 'idas', dae, t_eval[0],t_eval,opts)
#x0 = np.ones((nx,1))
tensor, x0 = initial(nx)
z0 = 0.0*x0
result = F(x0=x0, z0=z0)
x_res = result['xf'].full()
z_res = result['zf'].full()

X = x_res.transpose().reshape(nt,nx,9,4,4)
Z = z_res.transpose().reshape(nt,nx,9,4,4)

def check_solution(tensor, t, x_indices):
    for x in x_indices:
        print(f'Is it solved correctly at t = {t} and x = {x}:', np.allclose(tensor[x, :, :, :], X[t, x, :, :, :]))

# Initial conditions at different times

x_indices = [0,5,-1]
for t in t_eval.astype(int):
    tensor, _ = initial(nx, t=t)
    check_solution(tensor, t, x_indices)

plt.figure()