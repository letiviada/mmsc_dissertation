import numpy as np
from micro_compute import compute_results
from utils import save_micro_results
import argparse
import concurrent.futures

def compute_and_save(alpha, beta):
    G_initial = {
        (0, 2, 0, 1): 1, (2, 0, 0, -1): 1, (1, 3, 0, 1): 1, (3, 1, 0, -1): 1, 
        (0, 1, -1, 0): 1, (1, 0, 1, 0): 1, (2, 3, -1, 0): 1, (3, 2, 1, 0): 1,
        (0, 1, 0, 0): 1, (1, 0, 0, 0): 1, (0, 2, 0, 0): 1, (2, 0, 0, 0): 1, 
        (1, 3, 0, 0): 1, (3, 1, 0, 0): 1, (2, 3, 0, 0): 1, (3, 2, 0, 0): 1
    }
    tau_values = np.linspace(0, 10_000, 3001)
    results, time_passed = compute_results(alpha, beta, G_initial, tau_values, l=2.0)
    save_micro_results(alpha, beta, results, time_passed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs='+', type=float, help="List of alpha values")
    parser.add_argument("--betas", nargs='+', type=float, help="List of beta values")
    args = parser.parse_args()

    alphas = args.alphas
    betas = args.betas

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_and_save, alpha, beta) for alpha in alphas for beta in betas]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
