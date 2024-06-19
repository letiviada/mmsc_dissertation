import numpy as np
from casadi import *
from macro_solver import Solver
from utils import reshape_f, interp_functions, load_k_j
from plotting import plot_time, plot_one_dim, save_figure

class MultiscaleModel:
    def __init__(self, length=2.0, nt=6, nx=101):
        self.l = length
        self.nt = nt
        self.nx = nx
        self.t_eval = np.linspace(0, 1, self.nt)
        self.x_eval = np.linspace(0, self.l, self.nx)

    def load_and_interpolate(self):
        # Load data from the microscale model
        k, j, tau_eval = load_k_j()
        # Interpolate the data
        self.interp_k, self.interp_k_inv, self.interp_j = interp_functions(k, j, tau_eval)

    def setup_and_run(self):
        solv = Solver(self.l)
        # Setup the DAE solver
        F = solv.setup(self.interp_k, self.interp_k_inv, self.interp_j, self.t_eval, self.nx, self.l)
        # Run the DAE solver
        x_res, z_res = solv.run(F, self.nx)
        # Reshape the solution
        self.c, self.tau, self.u, self.psi = reshape_f(x_res, z_res, self.nt, self.nx)
    

def main():
    model = MultiscaleModel()
    model.load_and_interpolate()
    model.setup_and_run()
    # Plot the solution
    fig = plot_time(model.t_eval, model.x_eval, model.c, title='Concentration')
    save_figure(fig, 'multiscale/figures/concentration/c')
if __name__ == '__main__':
    main()