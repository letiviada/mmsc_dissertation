# Open Throughput model and for tthe beta values get predictions for a lot of alpha values. 
# Find the value of alpha closest to V for each beta value.
# Create a dataframe of opt values cleaned and this new endpoint
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import open_model, data_throughput, get_data_from_json, clean_data_throughput, save_data_to_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_dataset(throughput:float, save:bool = False) -> pd.DataFrame:

    total_throughput_model = open_model('total_throughput', model_path = 'regression/models_polynomial/')
    termination_time_model = open_model('termination_time', model_path = 'regression/models_polynomial/')
    efficiency_model = open_model('efficiency', model_path = 'regression/models_polynomial/')

    alpha = np.arange(0.0,1.001,0.001).round(4)
    beta = np.arange(0.01,0.11,0.01).round(3)
    alpha_beta_grid = np.array(np.meshgrid(alpha, beta)).T.reshape(-1, 2)
    inputs = pd.DataFrame(alpha_beta_grid, columns=['adhesivity', 'particle_size'])
    data_ml = pd.DataFrame({'adhesivity': inputs['adhesivity'], 'particle_size': inputs['particle_size']})

    throughput_predictions = total_throughput_model.predict(inputs)
    termination_time_predictions = termination_time_model.predict(inputs)
    efficiency_predictions = efficiency_model.predict(inputs)
    data_ml.loc[:, 'total_throughput'] = throughput_predictions[:]
    data_ml.loc[:, 'termination_time'] = termination_time_predictions[:]
    data_ml.loc[:, 'efficiency'] = efficiency_predictions[:]
    # Filter the data to get the values of alpha and beta that are closest to the throughput
    filtered_data = pd.DataFrame()
    for part_size in data_ml['particle_size'].unique():
        filtered_df = data_ml[data_ml['particle_size'] == part_size]
        filtered_df = filtered_df[filtered_df['total_throughput'] >= throughput]
        min_throughput = filtered_df['total_throughput'].min()
        optimum_values_for_size = filtered_df[filtered_df['total_throughput'] == min_throughput]

        if optimum_values_for_size['adhesivity'].values[0] != 1.0:
            filtered_data = pd.concat([filtered_data, optimum_values_for_size], ignore_index=True)
    # Get the data from the json file
    data = clean_data_throughput('performance_indicators/performance_indicators_phi_4.0.json', throughput)
    data_physical = data_throughput(throughput,data)
    data_physical = data_physical[['adhesivity', 'particle_size', 
                                   f'time_throughput_{throughput}', f'efficiency_throughput_{throughput}']]

    # Rename filtered data columns
    filtered_data.rename(columns={'termination_time': f'time_throughput_{throughput}', 
                                  'efficiency': f'efficiency_throughput_{throughput}'}, inplace=True) 
    final_data = pd.concat([data_physical, filtered_data], ignore_index=True)
    final_data['total_throughput'] = final_data['total_throughput'].fillna(100)
    if save == True:
        save_data_to_csv(final_data, f'optimization/opt_throughput/data/throughput_{throughput}', 'initial_dataset.csv')
    return final_data, inputs, total_throughput_model

if __name__ == '__main__':
    throughput = 100
    create_dataset(throughput, save=True)