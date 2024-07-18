import pandas as pd
from opt_get_models import get_full_data_and_models

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

def predict_minimum(data_predic:pd.DataFrame, output:str)-> pd.DataFrame:
    pass

if __name__ == '__main__':
    # Define the parameters
    filename = 'performance_indicators/performance_indicators_phi_1.0.json'
    time = 400
    n = 1
    outputs = ['volume_liquid', 'total_concentration']
    type_model = ['polynomial', 'gradient_boosting']

    # Get the data we have used to train the models
    full_data, vol_model, conc_model = get_full_data_and_models(outputs, time, n, type_model,filename)
    compare_prediction_of_minimum(full_data, 'ratio')


   
   

