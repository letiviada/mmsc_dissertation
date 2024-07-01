from utils import FilterPerformance
from utils import save_results
import argparse
import concurrent.futures
import numpy as np
import time
from utils import load_any

def performance_indicators(alpha,beta,phi,run,filename):
    t_eval = load_any(alpha,beta,'time_eval',run,filename)
    velocity = load_any(alpha,beta,'darcy_velocity',run,filename)
    concentration = load_any(alpha,beta,'concentration',run,filename)
    
    filter_performance = FilterPerformance(t_eval=t_eval, u=velocity,c=concentration)

    termination_time = filter_performance.termination_time(mu=0.1)
    time_ev = np.linspace(0,termination_time,101)
    throughput = filter_performance.throughput(tf=time_ev)
    efficiency = filter_performance.efficiency(t_eval=time_ev)
    lifetime = throughput[-1]
    performance_indicators={
            'time' : time_ev.tolist(),
            'termination_time': termination_time,
            'throughput':throughput.tolist(),
            'efficiency': efficiency.tolist(),
            'lifetime': lifetime
    }
    return performance_indicators

def compute(alpha, beta, phi,num_runs):
    results_all_runs = []
    # Check if we are running for mono-dispersed or poly-dispersed
    if num_runs > 1:
        filename = f'multiscale/results/poly-dispersed/macroscale/macro_results_phi_{phi}.json'
        directory='multiscale/results/poly-dispersed/performance_indicators'
    else:
        filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json'
        directory='multiscale/results/mono-dispersed/performance_indicators'
    
    for run in range(num_runs):
        start = time.time()
        end = time.time()
        time_passed = end - start
        perf_indicators = performance_indicators(alpha,beta,phi,run,filename)
        run_results = {
            'run': run,
            'time_passed': time_passed
        }
        run_results.update(perf_indicators)
        results_all_runs.append(run_results)
    save_results(alpha, beta, results_all_runs, scale = 'performance_indicators', phi = phi,directory=directory)
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

    with concurrent.futures.ProcessPoolExecutor(max_workers = 6) as executor:
        futures = [executor.submit(compute, alpha, beta,phi,num_run) for alpha in alphas for beta in betas for phi in phis]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")



if __name__ == '__main__':
    
    main()
    




