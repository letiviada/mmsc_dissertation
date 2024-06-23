import numpy as np
from micro_compute import compute_results
from utils import save_micro_results
import argparse

# Main function that computes the solution to the microscale model
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, help="Value of alpha")
    parser.add_argument("--beta", type=float, help="Value of beta")
    args = parser.parse_args()

    # Get alpha and beta values from command line arguments
    alpha = args.alpha
    beta = args.beta

    # Parameters of the model
    # -------------------
    G_initial = {(0, 2, 0, 1): 1, (2, 0, 0, -1): 1, (1, 3, 0, 1): 1, (3, 1, 0, -1): 1, 
     (0, 1, -1, 0): 1, (1, 0, 1, 0): 1, (2, 3, -1, 0): 1, (3, 2, 1, 0): 1,
       (0, 1, 0, 0): 1, (1, 0, 0, 0): 1, (0, 2, 0, 0): 1, (2, 0, 0, 0): 1, 
       (1, 3, 0, 0): 1, (3, 1, 0, 0): 1, (2, 3, 0, 0): 1, (3, 2, 0, 0): 1}
    
    tau_values = np.linspace(0,1000,101)
    results, time_passed = compute_results(alpha, beta, G_initial, tau_values, l=2.0)
    save_micro_results(alpha, beta, results, time_passed)

if __name__ == "__main__":
    main()

