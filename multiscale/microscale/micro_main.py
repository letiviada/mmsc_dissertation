
import numpy as np
from micro_compute import compute_results
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.utils import save_results
import argparse
import concurrent.futures

def compute_and_save(alpha, beta, num_runs):
    G_initial = {
        (0, 2, 0, 1): 1, (2, 0, 0, -1): 1, (1, 3, 0, 1): 1, (3, 1, 0, -1): 1, 
        (0, 1, -1, 0): 1, (1, 0, 1, 0): 1, (2, 3, -1, 0): 1, (3, 2, 1, 0): 1,
        (0, 1, 0, 0): 1, (1, 0, 0, 0): 1, (0, 2, 0, 0): 1, (2, 0, 0, 0): 1, 
        (1, 3, 0, 0): 1, (3, 1, 0, 0): 1, (2, 3, 0, 0): 1, (3, 2, 0, 0): 1
    }

    def poly_dispersed(mean,sigma):
        def generate_lognormal_value(mean, sigma):
            return np.random.lognormal(mean, sigma)
        processed_keys = set()
        for key in G_initial:
            if key not in processed_keys:
                # Generate a lognormally distributed value
                value = generate_lognormal_value(mean=0, sigma=1)
                G_initial[key] = value
                
                # Get the corresponding key (j, i, -r, -s)
                i, j, r, s = key
                corresponding_key = (j, i, -r, -s)
        
            if corresponding_key in G_initial:
                G_initial[corresponding_key] = value
                processed_keys.add(corresponding_key)
            
            # Mark this key as processed
            processed_keys.add(key)
        return G_initial

    #G_polydisperse = poly_dispersed()
    tau_values = np.linspace(0,1_000, 2_001)
    results_all_runs = []
    results_all_runs_with_G = []
    for run in range(num_runs):
        # Compute results
        if num_runs > 1:
            G_initial = poly_dispersed(mean=0.5,sigma=0.3)
        results_with_G, results, time_passed = compute_results(alpha, beta, G_initial, tau_values, l=1.0)

        run_results = {
            'run': run,
            'time_passed': time_passed
        }
        run_results.update(results)
        results_all_runs.append(run_results)
        results_all_runs_with_G.append(results_with_G)

    if num_runs > 1:
        save_results(alpha, beta, results_all_runs, scale = 'micro', directory='multiscale/results/poly-dispersed/microscale')
        save_results(alpha,beta, results_with_G,scale = 'micro',directory='multiscale/results/poly-dispersed/microscale/full_output')
    else:
        save_results(alpha, beta, results_all_runs, scale = 'micro', directory='multiscale/results/mono-dispersed/microscale')
        #save_results(alpha,beta, results_with_G,scale = 'micro',directory='multiscale/results/mono-dispersed/microscale/full_output')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs='+', type=float, help="List of alpha values")
    parser.add_argument("--betas", nargs='+', type=float, help="List of beta values")
    parser.add_argument("--num_runs", type=int, help="Number of runs to simulate")
    args = parser.parse_args()

    alphas = args.alphas
    betas = args.betas
    num_run = args.num_runs

    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(compute_and_save, alpha, beta, num_run) for alpha in alphas for beta in betas]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
