import json
import pandas as pd
import numpy as np
import os
from utils.help_functions import create_interp
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
    data = data_all[['adhesivity', 'particle_size']+ output]
    return data

def clean_data(filename: str) -> pd.DataFrame:
    data_to_keep = get_data_from_json(filename)
    
    data_to_keep = data_to_keep[data_to_keep['lifetime'] <= 200]
    return data_to_keep

def sampling_data(X, y, size):
    X_new = X.sample(size, random_state = 42)
    y_new = y.loc[X_new.index]

    return X_new, y_new

def data_time(time:int, names:list, data: pd.DataFrame) -> pd.DataFrame:
    """
    Function that creates a new column in the data for the time specified in the names columns

    Parameters:
    ----------
    time (int): the time we want to consider
    names (list): the list of names of the columns we want to consider
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new columns
    """
    #data = get_data_from_json(filename)
    filter_working_indices = data[data['termination_time'] > time].index
    filter_finished_indices = data[data['termination_time'] <= time].index
    for name in names:
        if name == 'volume_liquid':
            data.loc[filter_finished_indices,f'{name}_time_{time}'] = data.loc[filter_finished_indices, 'lifetime']
            y_axis = 'throughput'
        elif name == 'last_concentration':
            data.loc[filter_finished_indices, f'{name}_time_{time}'] = data.loc[filter_finished_indices, 'concentration_outlet'].apply(lambda x: x[-1])
            y_axis = 'concentration_outlet'

        for index in filter_working_indices:
            row = data.loc[index]
            interp_func = create_interp(row, y_axis)
            data.at[index, f'{name}_time_{time}'] = interp_func(time) if interp_func is not None else np.nan
    return data
def get_ratio(numerator: str, denominator: str, power: float, data: pd.DataFrame) -> pd.DataFrame:
    
    """
    Function that gets the ratio of two columns in the data

    Parameters:
    ----------
    numerator (str): the name of the column we want to consider as the numerator
    denominator (str): the name of the column we want to consider as the denominator
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new column
    """
    data['ratio'] = (data[numerator]) / (data[denominator] ** power)
    return data