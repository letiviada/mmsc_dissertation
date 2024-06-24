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

        results.append({
            'tau': tau,
            'G': G,
            'W': W,
            'delta': delta,
            'k': k,
            'j': j,
        })

        G_previous = G
    end = time.time()
    time_passed = end-start
    return results, time_passed

def save_results1(results,time_passed, filename='micro_results.json'):
    """
    Saves the results to a JSON file, converting NumPy arrays to lists.

    Parameters:
    results (list): List of dictionaries containing results for each s.
    filename (str): Name of the file to save results to.
    """
    # Initialize a dictionary to accumulate results by keys
    accumulated_results = {}
    for result in results:
        for key, value in result.items():
            if key not in accumulated_results:
                accumulated_results[key] = []
            if isinstance(value, np.ndarray):
                accumulated_results[key].append(value.tolist())
            else:
                accumulated_results[key].append(value)
    accumulated_results['time'] = time_passed
    
    # Save as JSON
    with open(filename, 'w') as f:
        json.dump(accumulated_results, f, indent=4)


#def load_results(filename='micro_results.json'):
    """
    Loads results from a JSON file and converts lists back to NumPy arrays.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    list: List of dictionaries with NumPy arrays where applicable.
    """
   # with open(filename, 'r') as f:
   #     results = json.load(f)
    
    # Convert lists back to NumPy arrays
   # for key, value in results.items():
   #     results[key] = np.array(value)
   # 
   # return results

