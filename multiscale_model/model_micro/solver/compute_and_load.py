import numpy as np
from solver.funct_def import solve_W, find_delta, find_k, find_j, solve_G, initial_G

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
    dtau = (tau_values[-1]-tau_values[0]) / (len(tau_values)-1)
    for tau in tau_values:
        if tau == 0:
            G = initial_G(G_initial)
        else:
            G = solve_G(alpha, beta, delta,G_previous, tau,dtau)
    
        W = solve_W(G,l)
        delta = find_delta(W,l)
        k = find_k(G,delta,l)
        j = find_j(alpha,G,delta,l)
        

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


def load_results(filename='results.npy'):
    """
    Loads the results from a file.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    list: List of dictionaries containing results for each s.
    """
    return np.load(filename, allow_pickle=True)

def extract_values(results, key):
    """
    Extracts values from the results for a specific key.

    Parameters:
    results (list): List of dictionaries containing results for each s.
    key (str): Key to extract values for.

    Returns:
    np.ndarray: Array of extracted values.
    """
    return np.array([result[key] for result in results])
