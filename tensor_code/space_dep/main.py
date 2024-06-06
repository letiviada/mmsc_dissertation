import numpy as np
import matplotlib.pyplot as plt
from model import initial
from solver import solver_dae, run, reshape_N4
from plotting.plots import plot_varying_time
from plotting.save import save_figure
from data.N4 import values_dict

def main():
    """
    Main function to run the DAE solver and plot the results.
    """
    cell_shape = (9,4,4)
    nx = 101
    nt = 6
    end_shape = (nt,nx,*cell_shape)
    t_eval = np.linspace(0, 5, nt) 
    x_eval = np.linspace(0, 1, nx) # Size of domain make general ?

    # Setup and run the integrator 
    F = solver_dae(cell_shape, x_eval, t_eval)
    _, x0 = initial(x_eval)
    z0 = 0.0 * x0
    x_sol_array, z_sol_array = run(F,x0, z0)
    # Reshape to required tensor form
    X = reshape_N4(end_shape,x_sol_array)
    Z = reshape_N4(end_shape,z_sol_array)

    # Plot the results
    fig = plot_varying_time(t_eval, x_eval, X, initial, values_dict)
    save_figure(fig,'tensor_code/space_dep/figures/conductance')
    #plt.show()
if __name__ == "__main__":
    main()
