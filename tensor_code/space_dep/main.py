import numpy as np
from model import initial
from solver import solver_dae, run_reshape_N4
from plotting import plot_results

def main():
    """
    Main function to run the DAE solver and plot the results.
    """
    array_length = 144
    nx = 101
    nt = 6
    t_eval = np.linspace(0, 5, nt)
    x_eval = np.linspace(0, 1, nx)

    # Setup and run the integrator
    F = solver_dae(array_length, x_eval, t_eval)
    _, x0 = initial(x_eval)
    z0 = 0.0 * x0
    X, Z = run_reshape_N4(F, x0, z0, nt, nx)

    # Plot the results
    plot_results(t_eval, x_eval, X, initial, save = True)
if __name__ == "__main__":
    main()
