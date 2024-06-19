import numpy as np
from micro_functions import *

def compute_results(alpha, beta,G_initial, tau_values,l):
    """
    Computes results for a range of s values.

    Parameters:
    G_initial (dictionary): Initial G given as a dictionary.
    s_values (np.ndarray): Array of s values to iterate over.

    Returns:
    list: List of dictionaries containing results for each s.
    """
    results = []
    for tau in tau_values:
        if tau == 0:
            G = initial_G(G_initial)
            #G = four_reg_prescribed(4,3)
        else:
            G = solve_G(alpha, beta, delta,G_previous, tau_values)
    
        W = solve_W(G,l)
        delta = find_delta(W,l)
        k = find_permeability(G,delta)
        j = find_adhesivity(alpha,G,delta,l)
    

        #print(f"s = {s}, G.shape = {G.shape}, W = {W}, delta.shape = {delta.shape}, k = {k}, j = {j}")

        results.append({
            'tau': tau,
            'G': G,
            'W': W,
            'delta': delta,
            'k': k,
            'j': j
        })

        G_previous = G

    return results

def save_results(results, filename='results.npy'):
    """
    Saves the results to a file.

    Parameters:
    results (list): List of dictionaries containing results for each s.
    filename (str): Name of the file to save results to.
    """
    np.save(filename, results, allow_pickle=True)

