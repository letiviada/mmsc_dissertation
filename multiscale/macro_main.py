import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from macro_solver import Solver
from utils import reshape_f, interp_functions, load_k_j, get_k_and_j
from plotting import plot_time, plot_one_dim, save_figure

class MultiscaleModel:
    def __init__(self, length=2.0, nt=501, nx=101):
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
    
    def obtain_k_and_j(self):
        self.k = get_k_and_j(self.tau, self.interp_k)
        self.j = get_k_and_j(self.tau, self.interp_j)
        return self.k, self.j
    

def main():
    model = MultiscaleModel()
    model.load_and_interpolate()
    model.setup_and_run()
    model.obtain_k_and_j()

    # Plot the solution
    # -----------------
    #fig = plot_time(model.t_eval, model.x_eval, model.c, title='Concentration')
    #fig2 = plot_time(model.t_eval, model.x_eval, model.tau, title='Auxiliar variable')
    #fig3 = plot_time(model.t_eval, model.x_eval, model.k, title='Permeability')
    #fig4 = plot_time(model.t_eval, model.x_eval, model.j, title='Adhesivity')
    fig5 = plot_one_dim(model.t_eval, [model.u], title='Darcy Velocity')
    #fig6 = plot_time(model.t_eval, model.x_eval,model.psi, title='Reactivity')
    fig7 = plot_one_dim(model.t_eval, [model.psi[:,1]], title='Reactivity')
    # Save the figure
    # ---------------
    #save_figure(fig, 'multiscale/figures/concentration/c')
    #save_figure(fig2, 'multiscale/figures/tau/tau')
    #save_figure(fig3, 'multiscale/figures/permeability/k')
    #save_figure(fig4, 'multiscale/figures/adhesivity/j')
    save_figure(fig5, 'multiscale/figures/velocity/u')
    #save_figure(fig6, 'multiscale/figures/reactivity/psi')
    save_figure(fig7, 'multiscale/figures/reactivity/psi_time')
if __name__ == '__main__':
    main()