import numpy as np
import pandas as pd
from utils_r.help_functions import create_interp
from scipy.integrate import quad

def data_throughput(throughput:float, data: pd.DataFrame) -> pd.DataFrame:
    """
    Function that creates a new column in the data for a given throughput

    Parameters:
    ----------
    throughput (fkoat): the total volume we want to output
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new columns
    """
    #data = get_data_from_json(filename)
    filter_working_indices = data[data['total_throughput'] >= throughput].index
    filter_finished_indices = data[data['total_throughput'] < throughput].index
    data.loc[filter_finished_indices, f'time_throughput_{throughput}'] = np.nan
    data.loc[filter_finished_indices, f'retained_particles_throughput_{throughput}'] = np.nan
    for index in filter_working_indices:
        row = data.loc[index]
        interp_func_time = create_interp(row, 'throughput','time')
        interp_func_flux_out = create_interp(row,'time', 'flux_out_time')
        time_throughput = interp_func_time(throughput) if interp_func_time is not None else np.nan
        data.at[index, f'time_throughput_{throughput}'] = time_throughput
        if row['adhesivity'] == 0.0:
            data.at[index, f'avg_retained_particles_throughput_{throughput}'] = 0.0
        else:
            data.at[index, f'avg_retained_particles_throughput_{throughput}'] = 1 - (interp_func_flux_out(time_throughput)) / throughput if interp_func_flux_out is not None else np.nan
    data_sorted = data.sort_values(['particle_size', 'adhesivity'])          
    return data_sorted

def data_efficiency(efficiency:float, data:pd.DataFrame) -> pd.DataFrame:
    """
    Function that creates a new column in the data for a given efficiency

    Parameters:
    ----------
    efficiency (float): the total volume we want to output
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new columns
    """
    filter_working_indices = data[data['efficiency'] >= efficiency].index
    filter_finished_indices = data[data['efficiency'] < efficiency].index
    data.loc[filter_finished_indices, f'time_efficiency_{efficiency}'] = np.nan
    data.loc[filter_finished_indices, f'volume_efficiency_{efficiency}'] = np.nan
    for index in filter_working_indices:
        row = data.loc[index]
        interp_func_time = create_interp(row, 'efficiency_time','time')
        interp_func_throughput = create_interp(row,'time', 'throughput')
        time_efficiency = interp_func_time(efficiency) if interp_func_time is not None else np.nan
        data.at[index, f'time_efficiency_{efficiency}'] = time_efficiency
        data.at[index, f'volume_efficiency_{efficiency}'] = interp_func_throughput(time_efficiency) if interp_func_throughput is not None else np.nan
    data_sorted = data.sort_values(['particle_size', 'adhesivity'])          
    return data_sorted

def data_time(time:int, data: pd.DataFrame) -> pd.DataFrame:
    """
    Function that creates a new column in the data for the time specified in the names columns

    Parameters:
    ----------
    time (int): the time we want to consider
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new columns
    """

    filter_working_indices = data[data['termination_time'] > time].index
    filter_finished_indices = data[data['termination_time'] <= time].index
    data.loc[filter_finished_indices,f'volume_time_{time}'] = data.loc[filter_finished_indices, 'lifetime']
    data.loc[filter_finished_indices, f'efficiency_time_{time}'] = data.loc[filter_finished_indices, 'efficiency'] 
    for index in filter_working_indices:
        row = data.loc[index]
        interp_func_efficiency = create_interp(row,'time','efficiency_time')
        interp_func_throughput = create_interp(row,'time', 'throughput')
        if row['adhesivity'] == 0.0:
                    data.at[index, f'efficiency_time_{time}'] = 0.0
        time_efficiency = interp_func_efficiency(time) if interp_func_efficiency is not None else np.nan
        data.at[index, f'efficiency_time_{time}'] = time_efficiency
        data.at[index, f'volume_time_{time}'] = interp_func_throughput(time) if interp_func_throughput is not None else np.nan
    data_sorted = data.sort_values(['particle_size', 'adhesivity'])  
    return data_sorted
