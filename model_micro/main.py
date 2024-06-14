import numpy as np
from solver.compute_and_load import compute_results, save_results, load_results, extract_values

def main():
    # Inputs of the model
    alpha, beta = 1, 1
    G_initial = {
        (0, 2, 1, 2): 1, (2, 0, 1, 0): 1, (1, 3, 1, 2): 1, (3, 1, 1, 0): 1,
        (0, 1, 0, 1): 1, (1, 0, 2, 1): 1, (2, 3, 0, 1): 1, (3, 2, 2, 1): 1,
        (0, 1, 1, 1): 1, (1, 0, 1, 1): 1, (0, 2, 1, 1): 1, (2, 0, 1, 1): 1,
        (1, 3, 1, 1): 1, (3, 1, 1, 1): 1, (2, 3, 1, 1): 1, (3, 2, 1, 1): 1
    }
    s_values = np.arange(0, 21)
    # COmpute and save results for all s values
    results = compute_results(alpha, beta, G_initial, s_values)
    save_results(results, 'model_micro/results.npy')
    # Load the results
    results = load_results('model_micro/results.npy')
    # Extract arrays for s and k
    s_array = extract_values(results, 's')
    k_array = extract_values(results, 'k')
    # Print the results
    print("s array:", s_array)
    print("k array:", k_array)

if __name__ == "__main__":
    main()

