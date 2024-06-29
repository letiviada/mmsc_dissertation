from utils import FilterPerformance
from utils import save_macro_results
import argparse
import concurrent.futures
import numpy as np
import time
from utils import load_any

def performance_indicators(alpha,beta,phi):
    # Load values
    start_time= time.time()
    filename = f'multiscale/results/macroscale/macro_results_phi_{phi}.json'
    t_eval = load_any(alpha,beta,'time_eval',filename)
    velocity = load_any(alpha,beta,'darcy_velocity',filename)
    concentration = load_any(alpha,beta,'concentration',filename)
    
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
    save_macro_results(alpha,beta,phi,performance_indicators,'multiscale/results/performance_indicators')
    end_time = time.time()
    time_passed = end_time - start_time
    print(f"Alpha: {alpha}, Beta: {beta},Phi: {phi} Time elapsed: {time_passed} seconds")

def main():
    parser = argparse.ArgumentParser(description='Run macro_main.py with parameters.')
    parser.add_argument("--alphas", nargs='+', type=float, help="List of alpha values")
    parser.add_argument("--betas", nargs='+', type=float, help="List of beta values")
    parser.add_argument("--phis", nargs='+', type=float, help="List of phi values")
    args = parser.parse_args()

    alphas = args.alphas
    betas = args.betas
    phis = args.phis

    with concurrent.futures.ProcessPoolExecutor(max_workers = 6) as executor:
        futures = [executor.submit(performance_indicators, alpha, beta,phi) for alpha in alphas for beta in betas for phi in phis]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")



if __name__ == '__main__':
    
    main()
    




