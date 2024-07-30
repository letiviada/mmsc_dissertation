import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import clean_data_efficiency, data_efficiency, save_data_to_csv, get_ratio
import pandas as pd
import numpy as np

def efficiency_model_varying_n(n_values, efficiency, data):
    all_data = pd.DataFrame()
    optimum_values = pd.DataFrame()
    for n in n_values:
        data_model= get_ratio(f'time_efficiency_{efficiency}',f'volume_efficiency_{efficiency}',n, data)
        data_model['n'] = n 
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['ratio'].max()
            optimum_values_for_size = filtered_df[filtered_df['ratio'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)
    data_no_n = data_model.drop(columns = ['n', 'ratio'])
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'n']]
    optimum_values.rename(columns={'adhesivity': f'adhesivity_efficiency_{efficiency}'}, inplace=True)
    save_data_to_csv(all_data,f'optimization/opt_efficiency/data/efficiency_{efficiency}', 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,f'optimization/opt_efficiency/data/efficiency_{efficiency}', 'optimum_values.csv')
    save_data_to_csv(data_no_n,f'optimization/opt_efficiency/data/efficiency_{efficiency}', 'data_for_sums.csv')
    return all_data, optimum_values, data_no_n

if __name__ == '__main__':
    efficiency = 0.2
    # Maybe we can add a minimum volume as well??
    # Varying the importance of time
    data = clean_data_efficiency('performance_indicators/performance_indicators_opt.json',efficiency)
    data_model = data_efficiency(efficiency,data)
    n_values = np.arange(0,2.05,0.01)
    efficiency_model_varying_n(n_values, efficiency, data_model) 