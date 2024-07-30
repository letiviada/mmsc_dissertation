import json
import pandas as pd
import numpy as np
import os
from utils.help_functions import create_interp
from scipy.integrate import quad
from scipy.stats.qmc import LatinHypercube

def sampling_data(X, y, size:int, method:str='random'):
    if method == 'random':
        X_new = X.sample(size)
        y_new = y.loc[X_new.index]
    elif method == 'latin_hypercube':
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        if size == 'all':
            size = 0.8 * X_np.shape[0]
        
        sampler = LatinHypercube(d = 2)
        lhs_samples = sampler.random(size)
        
    else:
        raise ValueError("Invalid sampling method. Please choose 'random' or 'latin_hypercube'.")

    return X_new, y_new

def data_time_old(time:int, names:list, data: pd.DataFrame) -> pd.DataFrame:
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
            for index in filter_working_indices:
                row = data.loc[index]
                interp_func = create_interp(row,'time', 'throughput')
                data.at[index, f'{name}_time_{time}'] = interp_func(time) if interp_func is not None else np.nan
        elif name == 'total_concentration':
            data.loc[filter_finished_indices, f'{name}_time_{time}'] = data.loc[filter_finished_indices, 'efficiency'] 
            # Calculate total concentrratiton processed
            for index in filter_working_indices:
                row = data.loc[index]
                interp_func = create_interp(row, 'time','efficiency_time')
                if row['adhesivity'] == 0.0:
                    data.at[index, f'{name}_time_{time}'] = 0.0
                else:
                    data.at[index, f'{name}_time_{time}'] = interp_func(time) if interp_func is not None else np.nan
    return data
