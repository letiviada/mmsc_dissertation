import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import get_data_from_json, data_time, save_data_to_csv, get_product, open_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(data,time):
    """
    Function that scales the data.

    Parameters:
    ----------
    data (pd.DataFrame): the data we want to scale

    Returns:
    -------
    data (pd.DataFrame): the scaled data
    """
    data.loc[:,f'volume_time_{time}_scaled'] = data.loc[:,f'volume_time_{time}'] / time
    return data

def time_model_varying_n(n_values, time, data, physical = True, ml = False):
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
        data_model_scaled = scale_data(data,time)
        #data_model_scaled = data
        #data_model_scaled.rename(columns={f'volume_time_{time}': f'volume_time_{time}_scaled'}, inplace=True)
        data_model= get_product(f'volume_time_{time}_scaled',f'efficiency_time_{time}',n, data_model_scaled)
        data_model['weight_coefficient'] = n 
        data_model.rename(columns={'product': 'gamma'}, inplace=True)
        all_data = pd.concat([all_data, data_model], ignore_index=True)
        for particle_size in data_model['particle_size'].unique():
            filtered_df = data_model[data_model['particle_size'] == particle_size]
            min_product = filtered_df['gamma'].max()
            optimum_values_for_size = filtered_df[filtered_df['gamma'] == min_product]
            optimum_values = pd.concat([optimum_values, optimum_values_for_size], ignore_index=True)
    # Create different dataframes       
    data_no_n = data_model.drop(columns = ['weight_coefficient', 'gamma'])
    optimum_values = optimum_values[['adhesivity', 'particle_size', 'weight_coefficient']]
    optimum_values.rename(columns={'adhesivity': f'adhesivity_time_{time}'}, inplace=True)
    data_for_gamma_model = all_data[['adhesivity', 'particle_size', 'weight_coefficient', 'gamma']]

    # Save data
    if physical == True:
        filepath = f'optimization/opt_time/data/physical'
    if ml == True:
        filepath = f'optimization/opt_time/data/ml'
    save_data_to_csv(all_data,filepath, 'data_varying_n_min.csv')  
    save_data_to_csv(optimum_values,filepath, 'optimum_values.csv')
    save_data_to_csv(data_no_n,filepath, 'data_for_sums.csv')
    save_data_to_csv(data_for_gamma_model,filepath, 'data_for_gamma_model.csv')
    return all_data, optimum_values, data_no_n

def open_all_models_get_predictions(time, alpha, beta):
    # Create the grid of inputs
    alpha_beta_grid = np.array(np.meshgrid(alpha, beta)).T.reshape(-1, 2)
    inputs = pd.DataFrame(alpha_beta_grid, columns=['adhesivity', 'particle_size'])
    data_ml = pd.DataFrame({'adhesivity': inputs['adhesivity'], 'particle_size': inputs['particle_size']})
    # Open models
    volume_model = open_model(f'volume_time_{time}', 'regression/models_polynomial/')
    efficiency_model = open_model(f'efficiency_time_{time}', 'regression/models_polynomial/')
    # Get predictions
    volume_predictions = volume_model.predict(inputs)

    efficiency_predictions = efficiency_model.predict(inputs)
    # Save predictions
    data_ml.loc[:, f'volume_time_{time}'] = volume_predictions[:]
    data_ml.loc[:, f'efficiency_time_{time}'] = efficiency_predictions[:]
    return data_ml

def main(time, physical, ml, alpha = None, beta = None):
    if physical == True:
        data = get_data_from_json('performance_indicators/performance_indicators_standard.json') 
        data_model = data_time(time,data) 
    if ml == True:
        data_model = open_all_models_get_predictions(time, alpha, beta)

    n_values = np.arange(0.0,15,0.25).round(3)
    time_model_varying_n(n_values, time, data_model, physical, ml)

if __name__ == '__main__':
    time = 400
    main(time, physical = True, ml = False)
    alpha = np.arange(0.0,1.001,0.01).round(2)
    beta = np.arange(0.01,0.11,0.01).round(3)
    main(time, physical = False, ml = True, alpha = alpha, beta = beta)
    

