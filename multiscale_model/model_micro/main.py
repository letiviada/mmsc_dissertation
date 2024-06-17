import numpy as np
from solver.compute_and_load import compute_results, save_results, load_results, extract_values

def main():
    # Inputs of the model
    alpha, beta = 1, 0.01
    G_initial = {
        (0, 2, 1, 2): 1, (2, 0, 1, 0): 1, (1, 3, 1, 2): 1, (3, 1, 1, 0): 1,
        (0, 1, 0, 1): 1, (1, 0, 2, 1): 1, (2, 3, 0, 1): 1, (3, 2, 2, 1): 1,
        (0, 1, 1, 1): 1, (1, 0, 1, 1): 1, (0, 2, 1, 1): 1, (2, 0, 1, 1): 1,
        (1, 3, 1, 1): 1, (3, 1, 1, 1): 1, (2, 3, 1, 1): 1, (3, 2, 1, 1): 1
    }
    tau_values = np.arange(0, 21)
    # Compute and save results for all s values
    results = compute_results(alpha, beta, G_initial, tau_values)
    save_results(results, 'multiscale_model/model_micro/results.npy')
    # Load the results
    results = load_results('multiscale_model/model_micro/results.npy')
    # Extract arrays for s and k
    tau_array = extract_values(results, 'tau')
    k_array = extract_values(results, 'k')
    j_array = extract_values(results, 'j')
    # Print the results
    #print("tau array:", tau_array)
    print("k array:", k_array)
    print("j array:", j_array)

if __name__ == "__main__":
    main()

