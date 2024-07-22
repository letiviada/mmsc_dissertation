import pandas as pd
import numpy as np
from opt_get_models import get_full_data_and_models, get_physical_model

def compare_prediction_of_minimum(full_data:pd.DataFrame, output:str)-> pd.DataFrame:
    """
    Function that gets the prediction of the minimum value of the ratio

    Parameters:
    -----------
    full_data (pd.DataFrame): the full dataframe with the predictions
    output (str): the output to be predicted

    Returns:
    --------
    rows_compared (pd.DataFrame): the rows with the minimum ratio and the predictions
    """
    min_ratio_index = full_data[output].idxmin()
    min_ratio_row = full_data.loc[[min_ratio_index]]
    name_pred= output.split('_time')[0] + '_predictions'
    min_ratio_predictions_index = full_data[name_pred].idxmin()
    min_ratio_predictions_row = full_data.loc[[min_ratio_predictions_index]]

    rows_compared = pd.concat([min_ratio_row, min_ratio_predictions_row])
    rows_compared.to_csv('/Users/letiviada/dissertation_mmsc/regression/optimization/comparison_min.csv', index=False)
    return rows_compared

def compare_prediction_of_maximum(full_data:pd.DataFrame, output:str)-> pd.DataFrame:
    """
    Function that gets the prediction of the maximum value of the ratio

    Parameters:
    -----------
    full_data (pd.DataFrame): the full dataframe with the predictions
    output (str): the output to be predicted

    Returns:
    --------
    rows_compared (pd.DataFrame): the rows with the maximum ratio and the predictions
    """
    max_ratio_index = full_data[output].idxmax()
    max_ratio_row = full_data.loc[[max_ratio_index]]
    name_pred= output.split('_time')[0] + '_predictions'
    max_ratio_predictions_index = full_data[name_pred].idxmax()
    max_ratio_predictions_row = full_data.loc[[max_ratio_predictions_index]]

    rows_compared = pd.concat([max_ratio_row, max_ratio_predictions_row])
    rows_compared.to_csv('/Users/letiviada/dissertation_mmsc/regression/optimization/comparison_max.csv', index=False)
    return rows_compared

def get_data_varying_n(particle_size_value, n_values,time, filename):
    data_for_n = pd.DataFrame()
    for n in n_values:
        data_for_this_n = pd.DataFrame()
        data_ratio = get_physical_model(['volume_liquid','total_concentration'],time,n,filename)
        data_ratio= data_ratio[data_ratio['particle_size'] == particle_size_value].sort_values('adhesivity')
        data_for_this_n[f'ratio_{n}'] = data_ratio['ratio']
        data_for_n = pd.concat([data_for_n,data_for_this_n], axis = 1)
    data_for_n = pd.concat([data_ratio['adhesivity'],data_ratio[f'volume_liquid_time_{time}'], data_ratio[f'total_concentration_time_{time}'],data_for_n], axis = 1) 
    data_for_n.to_csv(f'/Users/letiviada/dissertation_mmsc/regression/optimization/data_for_n_beta_{particle_size_value}.csv', index=False)   
    print('Data saved')
    return data_for_n

def get_data_varying_n(particle_sizes,n_values, time, filename):
    all_data = pd.DataFrame()
    # Assume get_physical_model now returns data for all particle sizes

    for n in n_values:
        data = get_physical_model(['volume_liquid', 'total_concentration'], time, n, filename)
        data_sorted = data.sort_values(['particle_size', 'adhesivity'])
        data_sorted[f'ratio_{n}'] = data_sorted['ratio']
        all_data = pd.concat([all_data, data_sorted[f'ratio_{n}']], axis=1)
    all_data = pd.concat([data_sorted[['particle_size', 'adhesivity', f'volume_liquid_time_{time}', f'total_concentration_time_{time}']],all_data], axis=1)

    # Group by particle size and save each group to a separate file
    for particle_size in particle_sizes:
        data_for_particle_size = pd.DataFrame()
        data_for_particle_size = all_data[all_data['particle_size'] == particle_size]
        data_for_particle_size.to_csv(f'/Users/letiviada/dissertation_mmsc/regression/optimization/data_for_n_beta_{particle_size}.csv', index=False)

    return all_data

if __name__ == '__main__':
    # Define the parameters
    filename = 'performance_indicators/performance_indicators_phi_1.0.json'
    time = 400
    n = 1
    outputs = ['volume_liquid', 'total_concentration']
    type_model = ['polynomial', 'gradient_boosting']

    # Get the data we have used to train the models
    #full_data, vol_model, conc_model = get_full_data_and_models(outputs, time, n, type_model,filename)
    #compare_prediction_of_minimum(full_data, 'ratio')

    # Get the data for the varying n
    n_values = np.arange(0.04,3.25,0.01).round(2)
    particle_sizes = np.linspace(0.03, 0.09, 7).round(2)
    particle_sizes = [0.08]
    data = get_data_varying_n(particle_sizes,n_values,time,filename)



   
   

