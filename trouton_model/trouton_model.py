import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

nx = 100
nt = 101
dx = 1 / (nx - 1)

def alg(x,z):
    dx = 1 / (nx - 1)
    alg_eqn = ca.SX(nx,1)
    alg_eqn[0] = z[0] - 1.0
    alg_eqn[-1] = z[-1] - 10.0
    term1 = (x[2:] + x[1:-1]) * (z[2:] - z[1:-1]) / (2 * dx ** 2)
    term2 = (x[1:-1] + x[:-2]) * (z[1:-1] - z[:-2]) / (2 * dx ** 2)
    alg_eqn[1:-1] = term1 - term2
    return alg_eqn

def ode(x,z):
    xz = x * z 
    dhdt=ca.SX(nx,1)
    #dhdt[0] = -(xz[1] - xz[0]) / dx
    #dhdt[0] = 0.0
    #dhdt[-1] = -(xz[-1] - xz[-2]) / dx
    dhdt[1:] = -(xz[1:] - xz[:-1]) / dx
    #for i in range(1,nx):
        #dhdt[i] = -(xz[i] - xz[i-1]) / (dx)
    return dhdt


x = ca.SX.sym('x',(nx,1))
z  = ca.SX.sym('z',(nx,1))
t_eval = np.linspace(0,5,nt)
opts = {'grid': t_eval, 'reltol':1e-6,'abstol':1e-6,'output_t0':True}
dae = {'x':x,'z':z, 'ode': ode(x,z), 'alg': alg(x,z)}
F = ca.integrator('F', 'idas', dae, opts)
x0 = np.ones((nx,1))
z0 = 0.0*np.ones((nx,1))
result = F(x0=x0, z0=z0)

x_res = result['xf'].full()
z_res = result['zf'].full()
#y_sol = np.concatenate(x_res,z_res)
x_eval = np.linspace(0,1,nx)
z_array = np.array(z_res)
x_array = np.array(z_res)
x_array2 = np.transpose(x_array)

plt.figure()
for i in range(0,nt,20):
    plt.plot(x_eval, x_array2[i,:], label =f'time {i} index')

plt.legend()
plt.show()

print(f'x is {x_res}')
print(f'z is {z_res}')