import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import open_model, change_name_time, ratio_predictions
from utils_r import get_data_from_json, data_time, obtain_data, make_data_frame, get_product
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
import pandas as pd

def ml_model(inputs:pd.DataFrame,output:str, time, type_model:str)->tuple:
    """
    Function that gets the ML model for a given output and time. It opens the saved model in pkl, predicts the 
    outputs and returns the inputs, outputs and predictions.

    Parameters:
    -----------
    inputs (pd.DataFrame): the inputs of the model (adhesivity and particle_size)
    output (str): the output to be predicted
    time (int): the time to be predicted
    type_model (str): the type of model to be used (polynomial, random_forest, gradient_boosting)

    Returns:
    --------
    predictions (np.array): the predictions of the model
    model (object): the model used for the predictions
    """
    if time is not None:
        output_name = change_name_time(output, time)
    elif time is None:
        output_name = output
    model_path = f'regression/models_{type_model}/'
    model = open_model(output = output_name, model_path = model_path)
    predictions = model.predict(inputs)
    return predictions, model

def get_physical_model_time(outputs:list,time:int,n:int,filename:str)->pd.DataFrame:
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
    physical_data = get_data_from_json(filename)
    data = data_time(time, outputs, physical_data)
    volume_name = change_name_time(outputs[0], time)
    concentration_name = change_name_time(outputs[1], time)
    data_model = obtain_data([volume_name, concentration_name], data)
    data_ratio = get_product(volume_name, concentration_name,n, data_model) 
    return data_ratio

def get_full_data_and_models(outputs:list, time: int, n:int, type_model: list,filename:str)->tuple:
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
    num_model (object): the model for the numerator
    den_model (object): the model for the denominator
    """
    df_ratio = get_physical_model_time(outputs, time,n,filename)
    inputs = df_ratio[['adhesivity', 'particle_size']]

    numerator_predictions,num_model = ml_model(inputs,outputs[0], time, type_model[0])
    denominator_predictions,den_model = ml_model(inputs,outputs[1], time,type_model[1])
    ratio = ratio_predictions(numerator_predictions, denominator_predictions, n)

    column_names = ['ratio_predictions'] +[f'{output}_predictions' for output in outputs]
    data_arrays = [ratio, numerator_predictions, denominator_predictions]
    cols = {column_name: array for column_name, array in zip(column_names, data_arrays)}
    full_data = make_data_frame(df_ratio, **cols)
    return full_data, num_model, den_model



