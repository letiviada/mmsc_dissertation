import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import clean_data_throughput, data_throughput, save_data_to_csv, get_product, open_model
import pandas as pd
import numpy as np

def scale_data(data,throughput):

    """
    Function that scales the data.

    Parameters:
    ----------
    data (pd.DataFrame): the data we want to scale
    throughput (float): the throughput we want to consider

    Returns:
    -------
    data (pd.DataFrame): the scaled data
    """
    data.loc[data['adhesivity'] == 0, 'time_throughput_100'] = 100
    data.loc[data['adhesivity'] == 0, 'efficiency_throughput_100'] = 0
    data.loc[:,f'time_throughput_{throughput}_scaled'] =  throughput / data.loc[:,f'time_throughput_{throughput}']
    return data

def throughput_model_varying_n(n_values, throughput, data, physical = True, ml = False):
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
        data_model_scaled = scale_data(data,throughput)
        data_model= get_product(f'time_throughput_{throughput}_scaled',f'efficiency_throughput_{throughput}',n, data_model_scaled)
        data_model['weight_coefficient'] = n 
        data_model.loc[:, f'time_throughput_{throughput}'] = data_model_scaled.loc[:, f'time_throughput_{throughput}']
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['product'].max()
            optimum_values_for_size = filtered_df[filtered_df['product'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)
    all_data.rename(columns={'product': f'gamma'}, inplace=True)
    data_model.rename(columns={'product': f'gamma'}, inplace=True)
    # Create different dataframes
    data_no_n = data_model.drop(columns = ['weight_coefficient', 'gamma'])
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'weight_coefficient']]
    optimum_values.rename(columns={'adhesivity': f'adhesivity_throughput_{throughput}'}, inplace=True)
    if physical == True:
        filepath = f'optimization/opt_throughput/data/physical'
    if ml == True:
        filepath = f'optimization/opt_throughput/data/ml'
    save_data_to_csv(all_data,filepath, 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,filepath, 'optimum_values.csv')
    save_data_to_csv(data_no_n,filepath, 'data_for_sums.csv')
    return all_data, optimum_values, data_no_n

def open_ml_models_get_more_data(data_ml,throughput, alpha, beta):
    """
    Function that opens the ml models and gets more data for the specific throughput.
    """

    # Alpha-beta grid before cleaning
    alpha_beta_grid = np.array(np.meshgrid(alpha, beta)).T.reshape(-1, 2)
    inputs_before = pd.DataFrame(alpha_beta_grid, columns=['adhesivity', 'particle_size'])
    time_model = open_model(f'time_throughput_{throughput}', model_path = 'regression/models_polynomial/')
    efficiency_model = open_model(f'efficiency_throughput_{throughput}', model_path = 'regression/models_polynomial/')

    data = pd.DataFrame()
    for particle_size in data_ml['particle_size'].unique():
        filtered_df = data_ml[data_ml['particle_size'] == particle_size]
        max_adhesivity = filtered_df['adhesivity'].max()
        inputs_size = inputs_before[inputs_before['particle_size'] == particle_size]
        inputs_size = inputs_size[inputs_size['adhesivity'] <= max_adhesivity]
        data= pd.concat([data, inputs_size], ignore_index=True)
    time_predictions = time_model.predict(data)
    efficiency_predictions = efficiency_model.predict(data)
    data.loc[:,f'time_throughput_{throughput}'] = time_predictions[:]
    data.loc[:,f'efficiency_throughput_{throughput}'] = efficiency_predictions[:]
    return data

if __name__ == '__main__':
    throughput = 100
    data = clean_data_throughput('performance_indicators/performance_indicators_opt.json',throughput) 
    # Ensures that the data can reach the necessary throughput
    data_model = data_throughput(throughput,data) # Gets the data for the specific throughput
    # Varying the importance of time
    n_values = np.arange(0.0,10.05,0.25).round(3)
    throughput_model_varying_n(n_values, throughput, data_model, physical = True, ml = False) 
    
    # Get data from the ML models
    pd.set_option('display.max_rows', None)
    alpha = np.arange(0.0,1.1,0.01).round(3)
    beta = np.arange(0.02,0.11,0.02).round(2)
    data_ml = pd.read_csv(f'regression/optimization/opt_throughput/data/throughput_{throughput}/initial_dataset.csv')
    data_forward = open_ml_models_get_more_data(data_ml,throughput, alpha, beta)

    # Varying the importance of efficiency
    n_values = np.arange(0.0,8.05,0.25).round(3)
    throughput_model_varying_n(n_values, throughput, data_forward, physical = False, ml = True)



