import argparse
import numpy as np
from casadi import *
from macro_solver import Solver
from utils import reshape_f, interp_functions, load_k_j, get_k_and_j, save_macro_results
import concurrent.futures
import time

class MultiscaleModel:
    def __init__(self,T=500,length=2.0, nt=51, nx=101):
        self.T = T
        self.l = length
        self.nt = nt
        self.nx = nx
        self.t_eval = np.linspace(0, self.T, self.nt)
        self.x_eval = np.linspace(0, self.l, self.nx)

    def load_and_interpolate(self,alpha,beta,phi):
        # Load data from the microscale model
        k, j, tau_eval = load_k_j(alpha,beta,phi)
        # Interpolate the data
        self.interp_k, self.interp_k_inv, self.interp_j = interp_functions(k, j, tau_eval)

    def setup_and_run(self,phi):
        solv = Solver(self.l)
        # Setup the DAE solver
        F = solv.setup(self.interp_k, self.interp_k_inv, self.interp_j, self.t_eval, self.nx, self.l,phi)
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

def compute_and_save(alpha, beta, phi):
    start = time.time()
    model = MultiscaleModel()
    model.load_and_interpolate(alpha, beta,phi)
    model.setup_and_run(phi)
    model.obtain_k_and_j()
    end = time.time()
    time_passed = end - start
    save_macro_results(alpha, beta,phi, model.output_dict(time_passed))
    print(f"Alpha: {alpha}, Beta: {beta},Phi: {phi} Time elapsed: {time_passed} seconds")
    

def main():
    #alpha,beta = 1.0, 1.0
    #phi = 0.9
    #compute_and_save(alpha, beta, phi)
    parser = argparse.ArgumentParser(description='Run macro_main.py with parameters.')
    parser.add_argument("--alphas", nargs='+', type=float, help="List of alpha values")
    parser.add_argument("--betas", nargs='+', type=float, help="List of beta values")
    parser.add_argument("--phis", nargs='+', type=float, help="List of phi values")
    args = parser.parse_args()

    alphas = args.alphas
    betas = args.betas
    phis = args.phis

    with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
        futures = [executor.submit(compute_and_save, alpha, beta,phi) for alpha in alphas for beta in betas for phi in phis]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
