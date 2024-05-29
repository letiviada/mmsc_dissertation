import numpy as np
import matplotlib.pyplot as plt
from solver import solver
from utils.plotting import plot_results

nx = 101
nt = 101
t_eval = np.linspace(0,5,nt)
x_eval = np.linspace(0,1,nx)
x_sol,z_sol = solver(nx,t_eval)

#plot_results(x_eval,t_eval,x_sol, option = 'time', variable = 'sheet thickness')
#plot_results(x_eval,t_eval,x_sol, option = 'space', variable='sheet thickness')
#plot_results(x_eval,t_eval,z_sol, option = 'time', variable = 'axial velocity')
#plot_results(x_eval,t_eval,z_sol, option = 'space', variable='axial velocity')

    # Create a new figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot thickness against time
plot_results(x_eval, t_eval, x_sol, option='time', variable='sheet thickness', ax=ax1, fig=fig)
    
    # Plot thickness against space
plot_results(x_eval, t_eval, x_sol, option='space', variable='sheet thickness', ax=ax2, fig=fig)
    
plt.show()

    # Create a new figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot thickness against time
plot_results(x_eval, t_eval, x_sol, option='time', variable='axial velocity', ax=ax1, fig=fig)
    
    # Plot thickness against space
plot_results(x_eval, t_eval, x_sol, option='space', variable='axial velocity', ax=ax2, fig=fig)
    
plt.show()