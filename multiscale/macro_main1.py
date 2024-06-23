import argparse
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from macro_solver import Solver
from utils import reshape_f, interp_functions, load_k_j, get_k_and_j, save_macro_results
#from plotting import plot_time, plot_one_dim, save_figure
import time

class MultiscaleModel:
    def __init__(self, T=450,length=2.0, nt=10, nx=101):
        self.T = T
        self.l = length
        self.nt = nt
        self.nx = nx
        self.t_eval = np.linspace(0, self.T, self.nt)
        self.x_eval = np.linspace(0, self.l, self.nx)

    def load_and_interpolate(self,alpha,beta):
        # Load data from the microscale model
        k, j, tau_eval = load_k_j(alpha,beta)
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
    def output_dict(self,time_passed):
        output_dict = {
        'concentration': self.c.tolist(),
        'auxiliar_variable': self.tau.tolist(),
        'permeability': self.k.tolist(),
        'adhesivity': self.j.tolist(),
        'darcy_velocity': self.u.tolist(),
        'reactivity': self.psi.tolist(),
        'time_eval': self.t_eval.tolist(),
        'x_eval': self.x_eval.tolist(),
        'time': time_passed}
        return output_dict
    

def main(alpha,beta):
    start = time.time()
    model = MultiscaleModel()
    model.load_and_interpolate(alpha,beta)
    model.setup_and_run()
    model.obtain_k_and_j()
    end = time.time() 
    time_passed = end-start
    save_macro_results(alpha,beta,model.output_dict(time_passed))
    print(f"Time elapsed: {end-start} seconds")
    
    # Plot solutions
    # -----------------
    #fig = plot_time(model.t_eval, model.x_eval, model.c, title='Concentration')
    #fig2 = plot_time(model.t_eval, model.x_eval, model.tau, title='Auxiliar variable')
    #fig3 = plot_time(model.t_eval, model.x_eval, model.k, title='Permeability')
    #fig4 = plot_time(model.t_eval, model.x_eval, model.j, title='Adhesivity')
    #fig5 = plot_one_dim(model.t_eval, [model.u], title='Darcy Velocity')
    #fig6 = plot_time(model.t_eval, model.x_eval,model.psi, title='Reactivity')
    #fig7 = plot_one_dim(model.t_eval, [model.psi[:,1]], title='Reactivity')

    # Save figures
    # ---------------
    #save_figure(fig, 'multiscale/figures/concentration/c')
    #save_figure(fig2, 'multiscale/figures/tau/tau')
    #save_figure(fig3, 'multiscale/figures/permeability/k')
    #save_figure(fig4, 'multiscale/figures/adhesivity/j')
    #save_figure(fig5, 'multiscale/figures/velocity/u')
    #save_figure(fig6, 'multiscale/figures/reactivity/psi')
    #save_figure(fig7, 'multiscale/figures/reactivity/psi_time')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run macro_main.py with parameters.')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha parameter value')
    parser.add_argument('--beta', type=float, required=True, help='Beta parameter value')

    args = parser.parse_args()
    main(args.alpha, args.beta)
