import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import get_data_from_json, data_time, save_data_to_csv, get_product
import pandas as pd
import numpy as np

def time_model_varying_n(n_values, time, data):
    """
    Function that takes a specific value of time and varies the importance of time. Finds the
    optimum adhesivity that maximises the product of volume and retained particles.

    Parameters:
    ----------
    n_values (np.array): the values of n we want to consider
    time (float): the time we want to consider
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    all_data (pd.DataFrame): the data with the new columns
    optimum_values (pd.DataFrame): the optimum values for the product
    data_no_n (pd.DataFrame): the data without the n column (used for the sums model)
    
    """
    all_data = pd.DataFrame()
    optimum_values = pd.DataFrame()
    for n in n_values:
        data_model= get_product(f'volume_time_{time}',f'efficiency_time_{time}',n, data)
        data_model['n'] = n 
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['product'].max()
            optimum_values_for_size = filtered_df[filtered_df['product'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)
    data_no_n = data_model.drop(columns = ['n', 'product'])
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'n']]
    optimum_values.rename(columns={'adhesivity': f'adhesivity_time_{time}'}, inplace=True)
    save_data_to_csv(all_data,f'optimization/opt_time/data/time_{time}', 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,f'optimization/opt_time/data/time_{time}', 'optimum_values.csv')
    save_data_to_csv(data_no_n,f'optimization/opt_time/data/time_{time}', 'data_for_sums.csv')
    return all_data, optimum_values, data_no_n

if __name__ == '__main__':
    time = 400
    data = get_data_from_json('performance_indicators/performance_indicators_opt.json') 
    data_model = data_time(time,data) 
    n_values = np.arange(0,10.05,0.01)
    time_model_varying_n(n_values, time, data_model) 