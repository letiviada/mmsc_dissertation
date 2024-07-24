import pandas as pd
import numpy as np
from opt_get_models import  get_physical_model_time
from utils import save_data_to_csv

def time_model_varying_n( n_values, time, filename):
    all_data = pd.DataFrame()
    optimum_values = pd.DataFrame()
    for n in n_values:
        data = get_physical_model_time(['volume_liquid', 'removed_particles'], time, n, filename)
        data_sorted = data.sort_values(['particle_size', 'adhesivity'])
        
        data_sorted['n'] = n 
        data_sorted.drop(columns = [f'volume_liquid_time_{time}', f'removed_particles_time_{time}'], inplace=True)
        all_data = pd.concat([all_data, data_sorted], ignore_index=True)
        for particle_size in data_sorted['particle_size'].unique():
            filtered_df = data_sorted[data_sorted['particle_size'] == particle_size]
            max_product = filtered_df['product'].max()
            optimum_values_for_size = filtered_df[filtered_df['product'] == max_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)

    
    all_data = all_data[['adhesivity', 'particle_size', 'n', 'product']]
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'n']]
    save_data_to_csv(all_data,'optimization', 'data_varying_n.csv')  
    save_data_to_csv(optimum_values,'optimization', 'optimum_values.csv')
    return all_data

if __name__ == '__main__':
    # Define the parameters
    filename = 'performance_indicators/performance_indicators_standard_basic.json'
    time = 400
    n = 1
    outputs = ['volume_liquid', 'removed_particles']
    type_model = ['polynomial', 'gradient_boosting']

    # Get the data for the varying n
    n_values = np.arange(0.01,3,0.01).round(2)
    data = time_model_varying_n(n_values,time,filename)



   
   

