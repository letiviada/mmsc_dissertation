import numpy as np
from micro_compute import compute_results, save_results
from utils import load_results, extract_values
from plotting import plot_time, plot_one_dim, save_figure

# Main function that computes the solution to the microscale model
def main():
    # Parameters of the model
    # -------------------
    alpha, beta = 1,1 

    G_initial = {(0, 2, 0, 1): 1, (2, 0, 0, -1): 1, (1, 3, 0, 1): 1, (3, 1, 0, -1): 1, 
     (0, 1, -1, 0): 1, (1, 0, 1, 0): 1, (2, 3, -1, 0): 1, (3, 2, 1, 0): 1,
       (0, 1, 0, 0): 1, (1, 0, 0, 0): 1, (0, 2, 0, 0): 1, (2, 0, 0, 0): 1, 
       (1, 3, 0, 0): 1, (3, 1, 0, 0): 1, (2, 3, 0, 0): 1, (3, 2, 0, 0): 1}
    tau_values = np.linspace(0,20,1001)


    # Compute and save results for all s values
    # -----------------------------------------
    results = compute_results(alpha, beta, G_initial, tau_values,l=2.0)
    save_results(results, 'multiscale/micro_results.npy')

    # Load the results
    # ----------------
    results = load_results('multiscale/micro_results.npy')

    # Extract arrays for s and k
    # --------------------------
    tau_array = extract_values(results, 'tau')
    k_array = extract_values(results, 'k')
    j_array = extract_values(results, 'j')

    # Print the results
    # ----------------
    #print("tau array:", tau_array)
    print("k array:", k_array)
    print("j array:", j_array)

    # Plot the results
    # ----------------
    functs = [k_array, j_array]
    fig = plot_one_dim(tau_values,functs,x_eval_type='tau',title='Permeability and Adhesivity')
    save_figure(fig,'multiscale/figures/microscale/permeability_and_adhesivity/k_j')

if __name__ == "__main__":
    main()

