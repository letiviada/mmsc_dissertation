import numpy as np
import matplotlib.pyplot as plt
from solver import solver
from utils.plotting import plot_results

nx = 101
nt = 101
t_eval = np.linspace(0,5,nt)
x_eval = np.linspace(0,1,nx)
x_sol,z_sol = solver(nx,t_eval)
# Exact initial solutions
sol_u_initial = lambda x: 9*x+1
sol_h_initial = lambda x: np.ones(x.shape)
# Solutions of the steady state
sol_u_final = lambda x: np.exp(np.log(10)*x)
sol_h_final = lambda x: 1/(np.exp(np.log(10)*x))

plt.style.use('seaborn-v0_8')
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8),dpi=300)   

plot_results(x_eval, t_eval, x_sol, exact_initial = sol_h_initial(x_eval), exact_infinity=sol_h_final(x_eval), option='time', variable='sheet thickness', ax=ax1, fig=fig1)
#plot_results(x_eval, t_eval, x_sol, option='space', variable='sheet thickness', ax=ax2, fig=fig1) 
#plt.show()

#fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_results(x_eval, t_eval, z_sol, exact_initial = sol_u_initial(x_eval), exact_infinity=sol_u_final(x_eval),option='time', variable='axial velocity', ax=ax2, fig=fig1)
#plot_results(x_eval, t_eval, z_sol, option='space', variable='axial velocity', ax=ax2, fig=fig2)    
#fig2.savefig('dae_examples/trouton_model/figures/axial_velocity.png')
fig1.savefig('dae_examples/trouton_model/figures/trouton_model.png')  
plt.show()