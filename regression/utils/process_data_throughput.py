import numpy as np
import pandas as pd
from utils.help_functions import create_interp
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
            data.at[index, f'avg_retained_particles_throughput_{throughput}'] = 1.0
        else:
            data.at[index, f'avg_retained_particles_throughput_{throughput}'] = (interp_func_flux_out(time_throughput)) / throughput if interp_func_flux_out is not None else np.nan
    data_sorted = data.sort_values(['particle_size', 'adhesivity'])          
    return data_sorted
