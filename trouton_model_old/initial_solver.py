import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def ode(x,z):
    dhdt=ca.SX(1,1)
    dhdt[0]=0.0
    return dhdt

def alg(x,z):
    alg_eqn = ca.SX(nx,1)
    alg_eqn[0] = z[0]-1
    alg_eqn[-1] = z[-1]-10
    for i in range(1, nx-1):
        alg_eqn[i] = (z[i+1] - 2 * z[i] + z[i-1]) / (dx **2)
    return alg_eqn

    
nx = 5
dx = 0.25
x = ca.SX.sym('x',(1,1))
z  = ca.SX.sym('z',(nx,1))
t_eval = np.linspace(0,1,11)
opts = {'grid': t_eval, 'reltol':1e-6,'abstol':1e-6,'output_t0':True}
dae = {'x':x,'z':z, 'ode': ode(x,z), 'alg': alg(x,z)}
F = ca.integrator('F', 'idas', dae, opts)
x0 = np.ones((1,1))
z0 = 0.0*np.ones((nx,1))
result = F(x0=x0, z0=z0)

x_res = result['xf'].full()
z_res = result['zf'].full()
#y_sol = np.concatenate(x_res,z_res)
print(f'x is {x_res}')
print(f'z is {z_res}')
#print(f'y_sol is {y_sol}')

x_eval = np.linspace(0,1,nx)

print(f' x shape is {x.shape}')
print(f'z shape is {z.shape}')

z_array = np.array(z_res)
print(f'z_res of 0 is {z_array[:,0]}')
plt.figure()
plt.plot(x_eval,z_array[:,0])
plt.show()
