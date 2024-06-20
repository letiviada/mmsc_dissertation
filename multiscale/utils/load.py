import numpy as np

def load_results(filename='multiscale/micro_results.npy'):
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
def load_k_j(filename='multiscale/micro_results.npy'):
    """
    Loads only the k and j values from the results file.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    tuple: Two numpy arrays containing k and j values.
    """
    results = load_results(filename)
    k_values = extract_values(results, 'k')
    j_values = extract_values(results,'j')
    tau_eval = extract_values(results,'tau')
    return k_values, j_values, tau_eval
