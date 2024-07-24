import json
import pandas as pd
import numpy as np
import os


def get_data_from_json(filename:str) -> pd.DataFrame:
    """
    Function that gets the json file and processes the data

    Parameters:
    ----------
    filename (str): the path to the json file

    Returns:
    -------
    ml_data (pd.DataFrame): the data in a pandas DataFrame
    """
    directory = '/home/viadacampos/Documents/mmsc_dissertation/multiscale/results/mono-dispersed/'
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        directory = '/Users/letiviada/dissertation_mmsc/multiscale/results/mono-dispersed/'
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("JSON file not found in any of the specified directories.")

    with open(filepath) as f:
        data_json = json.load(f)

    data_list = []
    for key, value in data_json.items():
        _, alpha_beta = key.split('=')
        keys_str = alpha_beta.strip("()").split(',')
        alpha,beta = float(keys_str[0]), float(keys_str[1])
        record = {'adhesivity': alpha, 'particle_size': beta}
        record.update(value)
        data_list.append(record)

    ml_data = pd.DataFrame(data_list)
    
    #data_to_keep = ml_data[['Adhesivity', 'Particle Size', 'Termination time', 'Lifetime']] 
    return ml_data

def obtain_data(output:list, data_all: pd.DataFrame) -> pd.DataFrame:
    data = data_all[['adhesivity', 'particle_size'] + output] 
    return data

def clean_data(filename: str) -> pd.DataFrame:
    data_to_keep = get_data_from_json(filename)
    
    data_to_keep = data_to_keep[data_to_keep['termination_time'] <= 2000]
    data_to_keep.rename(columns={'lifetime': 'total_throughput'}, inplace=True)
    return data_to_keep

def clean_data_throughput(filename: str,throughput:float) -> pd.DataFrame:
    """
    Function that cleans the data by removing the data with throughput lower than the specified value

    Parameters:
    ----------
    filename (str): the name of the json file
    throughput (float): the throughput value to be considered

    Returns:
    -------
    data_to_keep (pd.DataFrame): the data with the throughput higher than the specified
    
    """
    data_to_keep = get_data_from_json(filename)
    
    data_to_keep = data_to_keep[data_to_keep['lifetime'] >= throughput]
    data_to_keep.rename(columns={'lifetime': 'total_throughput'}, inplace=True)
    return data_to_keep