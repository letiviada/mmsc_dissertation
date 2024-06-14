import numpy as np
from casadi import *

array_length = 144
nx = 3
nt = 2
def alg(x,z):
    alg_eqn = z-0.5*x
    return alg_eqn

def ode(x,z):
    ode = - x * z
    return ode

def initial(nx=1, value = 1):
    tensor = np.zeros((9, 4, 4)) # Shape of the tensor (r,i,j)
    #tensor[0,:,:] = 1
    #tensor[1,:,:] = 2
    # Define initial values and their positions ()
    positions = [
        (1,0,2), (1,1,3),
        (3,0,1), (3,2,3),
        (4,0,1), (4,0,2), (4,1,0), (4,1,3), (4,2,0), (4,2,3), (4,3,1), (4,3,2),
        (5,1,0), (5,3,2),
        (7,2,0), (7,3,1)
    ]
    # Assign non-zero values to the tensor
    tensor[tuple(zip(*positions))] = value
    if nx== 1:
        initial_condition = tensor
    else:
        initial_condition = np.tile(tensor, (nx,1,1,1))

    in_cond = initial_condition.reshape(-1)
    return initial_condition, in_cond


x = SX.sym('x',(array_length*nx,1))
z = SX.sym('z',(array_length*nx,1))
t_eval = np.linspace(0,5,nt)

opts = {'reltol':1e-10,'abstol':1e-10}
dae = {'x':x,'z':z, 'ode': ode(x,z), 'alg': alg(x,z)}
F = integrator('F', 'idas', dae, t_eval[0],t_eval,opts)
#x0 = np.ones((nx,1))
tensor, x0 = initial(nx,value=1)
z0 = 0.0*x0
result = F(x0=x0, z0=z0)
x_res = result['xf'].full()
z_res = result['zf'].full()

X = x_res.transpose().reshape(nt,nx,9,4,4)
Z = z_res.transpose().reshape(nt,nx,9,4,4)
print(X[0,:,:,:])
print('Is the initial time ok:',np.allclose(tensor,X[0,:,:,:,:]))

exact = lambda t: 2/(t+2)
tensor2, _ = initial(nx,value = exact(5))
#print(f'The real solution is {tensor2}') 
#print(f'The approximate solution is  {X[-1,:,:,:]}')
print('Is it solved correctly for time t = 5:', np.allclose(tensor2, X[-1,:,:,:,:]))