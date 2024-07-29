import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import clean_data_throughput, data_throughput, save_data_to_csv, get_product
import pandas as pd
import numpy as np

def throughput_model_varying_n(n_values, throughput, data):
    all_data = pd.DataFrame()
    optimum_values = pd.DataFrame()
    for n in n_values:
        data_model= get_product(f'time_throughput_{throughput}', f'avg_retained_particles_throughput_{throughput}', n, data)
        data_model['n'] = n 
        #data_model.drop(columns = [f'time_throughput_{throughput}', f'avg_retained_particles_throughput_{throughput}'], inplace=True)
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['product'].min()
            optimum_values_for_size = filtered_df[filtered_df['product'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)

    #all_data = all_data[['adhesivity', 'particle_size', 'n', 'product']]
   #all_data = pd.concat([all_data, cols_to_drop], axis=1)
    data_no_n = data_model.drop(columns = ['n', 'product'])
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'n']]
    optimum_values.rename(columns={'adhesivity': f'adhesivity_throughput_{throughput}'}, inplace=True)
    save_data_to_csv(all_data,f'optimization/opt_throughput/data/throughput_{throughput}', 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,f'optimization/opt_throughput/data/throughput_{throughput}', 'optimum_values.csv')
    save_data_to_csv(data_no_n,f'optimization/opt_throughput/data/throughput_{throughput}', 'data_for_sums.csv')
    return all_data, optimum_values, data_no_n

if __name__ == '__main__':
    throughput = 100
    data = clean_data_throughput('performance_indicators/performance_indicators_opt.json',throughput) # Ensures that the data can reach the necessary throughpu
    data_model = data_throughput(throughput,data) # Gets the data for the specific throughput
    # Varying the importance of time
    n_values = np.arange(0,2.05,0.01)
    throughput_model_varying_n(n_values, throughput, data_model) 




