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
    json_filename = 'micro_results.json'
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

def load_k_j(alpha,beta,run,filename):
    """
    Loads only the k and j values from the results file.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    tuple: Two numpy arrays containing k and j values.
    """

    data = load_results(filename)
    if f'(alpha,beta)=({alpha},{beta})' not in data:
        raise KeyError(f'(alpha,beta)=({alpha},{beta}) not found in data')
    results = data[f'(alpha,beta)=({alpha},{beta})']
    if f'run={run}' not in results:
        raise KeyError(f'run = {run} not found in data')
    results_to_convert = results[f'run={run}']
    results_to_get = convert_to_numpy(results_to_convert)
    k_values = results_to_get['k']
    j_values = results_to_get['j']
    tau_eval = results_to_get['tau']
    return k_values, j_values, tau_eval

def load_any(alpha,beta,key,run=0,filename='multiscale/results/macro_results.json'):
    """
    Loads only the k and j values from the results file.

    Parameters:
    filename (str): Name of the file to load results from.

    Returns:
    tuple: Two numpy arrays containing k and j values.
    """
    data = load_results(filename)
    if f'(alpha,beta)=({alpha},{beta})' not in data:
        raise KeyError(f'(alpha,beta)=({alpha},{beta}) not found in data')
    results = data[f'(alpha,beta)=({alpha},{beta})']
    if f'run={run}' not in results:
        raise KeyError(f'run = {run} not found in data')
    results_to_convert = results[f'run={run}']
    results_to_get = convert_to_numpy(results_to_convert)
    values = results_to_get[key]
    return values

def save_results(alpha, beta,results_runs, scale, phi = None, directory='multiscale/results/microscale'):
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
    for result in results_runs:
        if not isinstance(result, dict):
            raise TypeError(f"Expected a dictionary but got {type(result)}: {result}")
        run_key = f'run={result["run"]}'
        accumulated_results[run_key] = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                        for key, value in result.items() if key!="run"}
    if not os.path.exists(directory):
        os.makedirs(directory)
    if scale == 'micro':
        filename = os.path.join(directory, f'micro_results_alpha_{alpha}_beta_{beta}.json')
    elif scale == 'macro':
        filename = os.path.join(directory, f'macro_results_alpha_{alpha}_beta_{beta}_phi_{phi}.json')
    elif scale == 'performance_indicators':
        filename = os.path.join(directory, f'performance_indicators_alpha_{alpha}_beta_{beta}_phi_{phi}.json')
    # Save as JSON
    with open(filename, 'w') as file:
        json.dump(accumulated_results, file, indent=4)
