import numpy as np
from casadi import *
import matplotlib.pyplot as plt

array_length = 144
nx = 101
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
    positions = np.array(list(values_dict.keys()))
    values = np.array(list(values_dict.values()))
    idx_nx = np.arange(nx)[:, None]
    idx_r = positions[:, 0]
    idx_i = positions[:, 1]
    idx_j = positions[:, 2]
    xi = x_axis[:, None]
    tensor_values = func(values,xi)

    tensor[idx_nx, idx_r, idx_i, idx_j] = tensor_values

    #for i, xi in enumerate(x_axis):
       # for pos, val in values_dict.items():
           # tensor[i, pos[0], pos[1], pos[2]] = func(val, xi =xi)

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

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(nrows=nt//3 + (nt%3>0), ncols=3,figsize = (15,8),dpi=300)
axes = axes.flatten()

for idx, t in enumerate(t_eval.astype(int)):
    tensor, _ = initial(nx, t=t)
    ax = axes[idx]
    l1a, = ax.plot(x_eval,X[t,:,1,0,2], color = 'k',label = "$G_{13}^{(0,1)^{T}}$",linewidth = 2)
    l1b, = ax.plot(x_eval,tensor[:,1,0,2], linestyle = '--', label = "Exact $G_{13}^{(0,1)^{T}}$",linewidth = 2)
    l2a, = ax.plot(x_eval,X[t,:,1,1,3], color = 'k',label = "$G_{24}^{(0,1)^{T}}$",linewidth = 2)
    l2b, = ax.plot(x_eval,tensor[:,1,1,3], linestyle = '--',label = "Exact $G_{24}^{(0,1)^{T}}$",linewidth = 2)
    l3a, = ax.plot(x_eval, X[t, :, 3, 0, 1], color = 'k',label='$G_{12}^{(-1,0)^{T}}$',linewidth = 2)
    l3b, = ax.plot(x_eval, tensor[:, 3, 0, 1], linestyle='--', label='Exact $G_{12}^{(-1,0)^{T}}$',linewidth = 2)
    ax.set_xlabel('x')
    ax.set_title(f't={t}',pad=2)
    ax.grid(True)

# Create a single legend
fig.legend(handles=[l1a,l1b,l2a,l2b, l3a,l3b], loc='upper center',bbox_to_anchor=(0.5, 1), ncol=6)
fig.savefig('tensor_code/examples/figures/conductance_easy.png')  
plt.tight_layout(rect=[0, 0.03, 1, 0.85])
plt.subplots_adjust(hspace=0.75, wspace=0.4) 
plt.show()
