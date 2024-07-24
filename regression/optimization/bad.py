import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import clean_data_throughput, create_interp
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import plot_optimum
import pandas as pd
import numpy as np
from scipy.integrate import quad
from utils import save_data_to_csv, get_product

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
        interp_func_efficiency = create_interp(row,'time', 'efficiency_time')
        time_throughput = interp_func_time(throughput) if interp_func_time is not None else np.nan
        data.at[index, f'time_throughput_{throughput}'] = time_throughput
        if row['adhesivity'] == 0.0:
            data.at[index, f'avg_retained_particles_throughput_{throughput}'] = 1.0
        else:
            data.at[index, f'avg_retained_particles_throughput_{throughput}'] = (quad(interp_func_efficiency, 0,time_throughput)[0]/ time_throughput) if interp_func_efficiency is not None else np.nan
    data_sorted = data.sort_values(['particle_size', 'adhesivity'])          
    return data_sorted

def throughput_model_varying_n(n_values, throughput, data):
    all_data = pd.DataFrame()
    optimum_values = pd.DataFrame()
    for n in n_values:
        data_model = get_product(f'time_throughput_{throughput}', f'avg_retained_particles_throughput_{throughput}', n, data)
        data_model['n'] = n 
        data_model.drop(columns = [f'time_throughput_{throughput}', f'avg_retained_particles_throughput_{throughput}'], inplace=True)
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['product'].min()
            optimum_values_for_size = filtered_df[filtered_df['product'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)

    
    all_data = all_data[['adhesivity', 'particle_size', 'n', 'product']]
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'n']]
    save_data_to_csv(all_data,'optimization', 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,'optimization', 'optimum_values_min.csv')
    return all_data

data = clean_data_throughput('performance_indicators/performance_indicators_standard_basic.json',100)
unique_adhesivity_values = data['adhesivity'].unique()
print(unique_adhesivity_values)
data_ml = data_throughput(100,data)
pd.set_option('display.max_columns', None)
n_values = [1,2,3,4,5]
dat = throughput_model_varying_n(n_values, 100, data_ml)
plot_optimum([0.08],dat)