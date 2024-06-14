import numpy as np
from casadi import * 

nx = 144
nt = 6

def alg(x,z):
    alg_eqn = z-0.5*x
    return alg_eqn

def ode(x,z):
    ode = - x * z
    return ode
def initial(t=0):
    tensor = np.zeros((9, 4, 4))

    def func(t, value):
        c = 2 / value
        return 2 / (t + c)
    
    values_dict = {
    (1, 0, 2): 1, (1, 1, 3): 2,
    (3, 0, 1): 3, (3, 2, 3): 4,
    (4, 0, 1): 5, (4, 0, 2): 6, (4, 1, 0): 5, (4, 1, 3): 7, (4, 2, 0): 6, (4, 2, 3): 8, (4, 3, 1): 7, (4, 3, 2): 8,

    (5, 1, 0): 9, (5, 3, 2): 10,
    (7, 2, 0): 11, (7, 3, 1): 12
    }
    for pos, val in values_dict.items():
        tensor[pos] = func(t, val)
    initial_condition = tensor.reshape(-1)
    
    return tensor, initial_condition


def initial2():
    tensor = np.zeros((9, 4, 4)) # Shape of the tensor (r,i,j)

    # Define initial values and their positions ()
    positions = {
        (1,0,2):1, (1,1,3):2,
        (3,0,1):3, (3,2,3):4,
        (4,0,1):30, (4,0,2):6, (4,1,0):5, (4,1,3):7, (4,2,0):6, (4,2,3):8, (4,3,1):7, (4,3,2):8,
        (5,1,0):9, (5,3,2):10,
        (7,2,0):11, (7,3,1):12
    }
    # Assign non-zero values to the tensor
    for pos, val in positions.items():
        tensor[pos] = val
    initial_condition  = tensor.reshape(-1) # Reshape into a 1d array (it does so by rows)
    return tensor, initial_condition
tensor,_ = initial2()
print(tensor)
x = SX.sym('x',(nx,1))
z = SX.sym('z',(nx,1))
t_eval = np.linspace(0,5,nt)

opts = {'reltol':1e-10,'abstol':1e-10}
dae = {'x':x,'z':z, 'ode': ode(x,z), 'alg': alg(x,z)}
F = integrator('F', 'idas', dae, t_eval[0],t_eval,opts)
#x0 = np.ones((nx,1))
tensor, x0 = initial()
z0 = 0.0*x0
result = F(x0=x0, z0=z0)
x_res = result['xf'].full()
z_res = result['zf'].full()

X = x_res.transpose().reshape(nt,9,4,4)
Z = z_res.transpose().reshape(nt,9,4,4)
#print(X[0,:,:,:])
#print(tensor)
print('Is the initial time ok:',np.allclose(tensor,X[0,:,:,:]))

#exact = lambda a,t: 2/(t+a)
tensor2, _ = initial(t=5)
#print(f'The real solution is {tensor2}') 
#print(f'The approximate solution is  {X[-1,:,:,:]}')
print('Is it solved correctly for time t = 5:', np.allclose(tensor2, X[-1,:,:,:]))