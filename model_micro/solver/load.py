import numpy as np

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

if __name__ == "__main__":
    results = load_results('model_micro/results.npy')

    # Extract arrays for s and k
    s_array = extract_values(results, 's')
    k_array = extract_values(results, 'k')

    print("s array:", s_array)
    print("k array:", k_array)
