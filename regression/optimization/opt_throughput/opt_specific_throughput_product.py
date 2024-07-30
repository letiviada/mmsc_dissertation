import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import clean_data_throughput, data_throughput, save_data_to_csv, get_ratio
import pandas as pd
import numpy as np

def throughput_model_varying_n(n_values, throughput, data):
    """
    Function that takes a specific value of throughput and varies the importance of time. Finds the 
    optimum adhesivity for the product of retained particles and the time.

    Parameters:
    ----------
    n_values (np.array): the values of n we want to consider
    throughput (float): the throughput we want to consider
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
        data_model= get_ratio(f'avg_retained_particles_throughput_{throughput}',f'time_throughput_{throughput}',n, data)
        data_model['n'] = n 
        #data_model.drop(columns = [f'time_throughput_{throughput}', f'avg_retained_particles_throughput_{throughput}'], inplace=True)
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['ratio'].max()
            optimum_values_for_size = filtered_df[filtered_df['ratio'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)
    data_no_n = data_model.drop(columns = ['n', 'ratio'])
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'n']]
    optimum_values.rename(columns={'adhesivity': f'adhesivity_throughput_{throughput}'}, inplace=True)
    save_data_to_csv(all_data,f'optimization/opt_throughput/data/throughput_{throughput}', 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,f'optimization/opt_throughput/data/throughput_{throughput}', 'optimum_values.csv')
    save_data_to_csv(data_no_n,f'optimization/opt_throughput/data/throughput_{throughput}', 'data_for_sums.csv')
    return all_data, optimum_values, data_no_n

if __name__ == '__main__':
    throughput = 100
    data = clean_data_throughput('performance_indicators/performance_indicators_opt.json',throughput) 
    # Ensures that the data can reach the necessary throughpu
    data_model = data_throughput(throughput,data) # Gets the data for the specific throughput
    # Varying the importance of time
    n_values = np.arange(0,10.05,0.01)
    throughput_model_varying_n(n_values, throughput, data_model) 




