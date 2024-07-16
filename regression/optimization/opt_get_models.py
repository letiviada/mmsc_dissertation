import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import open_model, change_name_time, open_data_model
from utils import get_data_from_json, data_time, obtain_data, get_ratio, make_data_frame
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import opt_ml
import matplotlib.pyplot as plt
import pandas as pd
def ml_model(output:str, time:int, type_model:str)->tuple:
    """
    Function that gets the ML model for a given output and time. It opens the saved model in pkl, predicts the 
    outputs and returns the inputs, outputs and predictions.

    Parameters:
    -----------
    output (str): the output to be predicted
    time (int): the time to be predicted
    type_model (str): the type of model to be used (polynomial, random_forest, gradient_boosting)

    Returns:
    --------
    inputs (pd.DataFrame): the inputs of the model (adhesivity and particle_size)
    outputs (pd.Series): the outputs of the model
    predictions (np.array): the predictions of the model
    """
    output_name = change_name_time(output, time)
    model_path = f'regression/models_{type_model}/'
    model = open_model(output = output_name, model_path = model_path)
    inputs, outputs = open_data_model('total', output_name, model_path = model_path)
    predictions = model.predict(inputs)
    return predictions

def get_physical_model(outputs:list,time:int)->tuple:
    """
    Function that gets the models from the ML and Pjysical results for a given output and time.

    Parameters:
    -----------
    outputs (list): the outputs to be predicted
    time (int): the time to be predicted

    Returns:
    --------
    data_ratio (pd.DataFrame): the ratio of the physical model
    """
    physical_data = get_data_from_json('performance_indicators/performance_indicators_standard.json')
    data = data_time(time, outputs, physical_data)
    volume_name = change_name_time(outputs[0], time)
    concentration_name = change_name_time(outputs[1], time)
    data_model = obtain_data([volume_name, concentration_name], data)
    data_ratio = get_ratio(volume_name, concentration_name,n, data_model) 
    return data_ratio

def ratio_predictions(numerator_predictions, denominator_predictions, n):
    """
    Function that calculates the ratio predictions

    Parameters:
    -----------
    volume_predictions (np.array): the volume predictions
    concentration_predictions (np.array): the concentration predictions
    n (int): the power to which the volume predictions are raised

    Returns:
    --------
    ratio (np.array): the ratio predictions
    """
    return (numerator_predictions ** n)/(denominator_predictions)

def get_full_data(outputs:list, time: int, n:int, type_model: list):
    """
    Function that getsthefull dataframe with the predictions for the numerrtor, denominator and ratio

    Parameters:
    -----------
    outputs (list): the outputs to be predicted
    time (int): the time to be predicted
    n (int): the power to which the volume predictions are raised
    type_model (list): the type of model to be used (polynomial, random_forest, gradient_boosting)

    Returns:
    --------
    full_data (pd.DataFrame): the dataframe with the predictions
    """
    numerator_predictions = ml_model(outputs[0], time, type_model[0])
    denominator_predictions = ml_model(outputs[1], time,type_model[1])
    ratio = ratio_predictions(numerator_predictions, denominator_predictions, n)
    df_ratio = get_physical_model(outputs, time)
    column_names= ['ratio_predictions'] +[f'{output}_predictions' for output in outputs]
    data_arrays = [ratio, numerator_predictions, denominator_predictions]
    cols = {column_name: array for column_name, array in zip(column_names, data_arrays)}
    full_data = make_data_frame(df_ratio, **cols)
    return full_data

def make_plots(full_data:pd.DataFrame, lines: bool, actual: bool, prediction: bool):
    opt_ml(full_data, 'ratio', lines, actual, prediction)


if __name__ == '__main__':
    time = 400
    n = 1
    outputs = ['volume_liquid', 'total_concentration']
    type_model = ['polynomial', 'gradient_boosting']
    full_data = get_full_data(outputs, time, n, type_model)
    make_plots(full_data, lines = False, actual = True, prediction = True)
