import json
import zipfile
import os
import numpy as np


def load_results(filename='multiscale/results/microscale/micro_results.json'):
    """
    Loads results from a JSON file and converts lists back to NumPy arrays.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    list: List of dictionaries with NumPy arrays where applicable.
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results

def load_zip(filename = 'multiscale/results/microscale/micro_results.zip'):
    zip_path = filename
    json_filename = 'multiscale/results/microscale/micro_results.json'
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract the JSON file from the zip archive
        with zip_ref.open(json_filename) as json_file:
            results = json.load(json_file)
    return results

def convert_to_numpy(results):
        
    # Convert lists back to NumPy arrays
    for key, value in results.items():
        results[key] = np.array(value)
    return results

def load_k_j(alpha,beta,phi,filename='multiscale/results/microscale/micro_results.zip'):
    """
    Loads only the k and j values from the results file.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    tuple: Two numpy arrays containing k and j values.
    """

    data = load_zip(filename)
    if f'(alpha,beta)=({alpha},{beta})' not in data:
        raise KeyError(f'(alpha,beta)=({alpha},{beta}) not found in data')
    results = data[f'(alpha,beta)=({alpha},{beta})']
    results = convert_to_numpy(results)
    k_values = results['k']
    j_values = results['j']
    tau_eval = results['tau']
    return k_values, j_values, tau_eval

def load_any(alpha,beta,phi,key,filename='multiscale/results/macro_results.json'):
    """
    Loads only the k and j values from the results file.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    tuple: Two numpy arrays containing k and j values.
    """
    data = load_results(filename)
    if f'(alpha,beta,phi)=({alpha},{beta},{phi})' not in data:
        raise KeyError(f'(alpha,beta)=({alpha},{beta}) not found in data')
    results = data[f'(alpha,beta,phi)=({alpha},{beta},{phi})']
    results = convert_to_numpy(results)
    values = results[key]
    return values

def save_macro_results(alpha, beta,phi, output_dict, directory='multiscale/results/macroscale'):
    """
    Saves the results to a JSON file, converting NumPy arrays to lists.

    Parameters:
    alpha (float): Adhesivity
    beta (float): Particle size
    output_dict (dict): Dictionary containing the output values of the macroscale model.
    directory (str): Directory to save the results to.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate the filename dynamically based on alpha and beta values
    filename = os.path.join(directory, f'macro_results_alpha_{alpha}_beta_{beta}_phi_{phi}.json')

    # Save as JSON
    with open(filename, 'w') as file:
        json.dump(output_dict, file, indent=4)


def save_micro_results(alpha, beta,results, time_passed, directory='multiscale/results/microscale'):
    """
    Saves the results to a JSON file, converting NumPy arrays to lists.

    Parameters:
    alpha (float): Alpha value.
    beta (float): Beta value.
    results (dict): Dictionary containing the output values of the microscale model.
    time_passed (float): Time passed during the computation.
    directory (str): Directory to save the results to.
    """
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

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate the filename dynamically based on alpha and beta values
    filename = os.path.join(directory, f'micro_results_alpha_{alpha}_beta_{beta}.json')

    # Save as JSON
    with open(filename, 'w') as file:
        json.dump(accumulated_results, file, indent=4)
