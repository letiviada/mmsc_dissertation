import argparse
import numpy as np
from casadi import *
from macro_solver import Solver
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.utils import reshape_f, interp_functions, load_k_j, get_k_and_j, save_results
import concurrent.futures
import time

class MultiscaleModel:
    def __init__(self,T=1500,length=2.0, nt=301, nx=151):
        self.T = T
        self.l = length
        self.nt = nt
        self.nx = nx
        self.t_eval = np.linspace(0, self.T, self.nt)
        self.x_eval = np.linspace(0, self.l, self.nx)

    def load_and_interpolate(self,alpha,beta,run,filename):
        # Load data from the microscale model
        k, j, tau_eval = load_k_j(alpha,beta,run,filename)
        # Interpolate the data
        self.interp_k, self.interp_k_inv, self.interp_j = interp_functions(k, j, tau_eval)

    def setup_and_run(self,phi):
        phi = 1
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
    def output_dict(self):
        output_dict = {
        'concentration': self.c.tolist(),
        'auxiliar_variable': self.tau.tolist(),
        'permeability': self.k.tolist(),
        'adhesivity': self.j.tolist(),
        'darcy_velocity': self.u.tolist(),
        'reactivity': self.psi.tolist(),
        'time_eval': self.t_eval.tolist(),
        'x_eval': self.x_eval.tolist()
        }
        return output_dict

def compute_and_save(alpha, beta, phi, num_run):
    results_all_runs = []
    # Check if we are running for mono-dispersed or poly-dispersed
    if num_run > 1:
        filename = f'multiscale/results/poly-dispersed/microscale/micro_results.json'
        directory='multiscale/results/poly-dispersed/macroscale'
    else:
        filename = f'multiscale/results/mono-dispersed/microscale/micro_results.json'
        directory='multiscale/results/mono-dispersed/macroscale'
    for run in range(num_run):
        actual_run = None if num_run ==1 else run
        start = time.time()
        model = MultiscaleModel()
        model.load_and_interpolate(alpha, beta, actual_run,filename)
        model.setup_and_run(phi)
        model.obtain_k_and_j()
        end = time.time()
        time_passed = end - start
        run_results = {
            'run': actual_run,
            'time_passed': time_passed
        }
        run_results.update(model.output_dict())
        results_all_runs.append(run_results)

    # Save the results in a dictionary
    save_results(alpha, beta, results_all_runs, scale = 'macro',  phi = phi,directory=directory)
    print(f"Alpha: {alpha}, Beta: {beta}, Phi: {phi} Time elapsed: {time_passed} seconds")
    

def main():
    parser = argparse.ArgumentParser(description='Run macro_main.py with parameters.')
    parser.add_argument("--alphas", nargs='+', type=float, help="List of alpha values")
    parser.add_argument("--betas", nargs='+', type=float, help="List of beta values")
    parser.add_argument("--phis", nargs='+', type=float, help="List of phi values")
    parser.add_argument("--num_runs", type = int, help = "Number of runs to simulate")

    args = parser.parse_args()

    alphas = args.alphas
    betas = args.betas
    phis = args.phis
    num_run = args.num_runs
    with concurrent.futures.ProcessPoolExecutor(max_workers = 7) as executor:
        futures = [executor.submit(compute_and_save, alpha, beta,phi, num_run) for alpha in alphas for beta in betas for phi in phis]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
