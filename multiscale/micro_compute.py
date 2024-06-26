import json
import numpy as np
import time
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
    start = time.time()
    results = []
    results_with_G = []
    for tau in tau_values:
        if tau == 0:
            G = initial_G(G_initial)
        else:
            G = solve_G(alpha, beta, delta,G_previous, tau_values)
    
        W = solve_W(G,l)
        delta = find_delta(W,l)
        k = find_permeability(G,delta,l)
        j = find_adhesivity(alpha,G,delta,l)

        #print(f"s = {s}, G.shape = {G.shape}, W = {W}, delta.shape = {delta.shape}, k = {k}, j = {j}")

        results_with_G.append({
            'tau': tau,
            'G': G,
            'W': W,
            'delta': delta,
            'k': k,
            'j': j,
        })
        results.append({
            'tau': tau,
            'k': k,
            'j': j,
        })

        G_previous = G
    end = time.time()
    time_passed = end-start
    return results_with_G,results, time_passed
