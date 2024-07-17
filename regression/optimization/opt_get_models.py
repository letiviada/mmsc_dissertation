import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import open_model, change_name_time, open_data_model
from utils import get_data_from_json, data_time, obtain_data, get_ratio, make_data_frame
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import opt_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
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

def get_physical_model(outputs:list,time:int,filename:str)->tuple:
    """
    Function that gets the models from the ML and Pjysical results for a given output and time.

    Parameters:
    -----------
    outputs (list): the outputs to be predicted
    time (int): the time to be predicted

    Returns:
    --------
    data_ratio (pd.DataFrame): the ratio of the physical model
    data_model (object): the data from the ML model
    """
    #physical_data = get_data_from_json()
    physical_data = get_data_from_json(filename)
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

def get_full_data(outputs:list, time: int, n:int, type_model: list,filename:str):
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
    df_ratio = get_physical_model(outputs, time,filename)
    inputs = df_ratio[['adhesivity', 'particle_size']]

    numerator_predictions,num_model = ml_model(inputs,outputs[0], time, type_model[0])
    denominator_predictions,den_model = ml_model(inputs,outputs[1], time,type_model[1])
    ratio = ratio_predictions(numerator_predictions, denominator_predictions, n)

    column_names= ['ratio_predictions'] +[f'{output}_predictions' for output in outputs]
    data_arrays = [ratio, numerator_predictions, denominator_predictions]
    cols = {column_name: array for column_name, array in zip(column_names, data_arrays)}
    full_data = make_data_frame(df_ratio, **cols)
    return full_data, num_model, den_model

def get_data_for_opt(models: list, names_models: list, input1: np.ndarray, input2: np.ndarray, ratio:bool,n:float)->pd.DataFrame:
    """
    Function that retrieves the data for plotting from the given models.

    Parameters:
    -----------
    models (list): the list of models to retrieve data from

    Returns:
    --------
    data (pd.DataFrame): the data for plotting
    """
    alpha_beta_grid = np.array(np.meshgrid(input1, input2)).T.reshape(-1, 2)
    inputs = pd.DataFrame(alpha_beta_grid, columns=['adhesivity', 'particle_size'])
    data = pd.DataFrame()
    for i,model in enumerate(models):
        prediction = model.predict(inputs)
        output_name = f'{names_models[i]}_predictions'
        name_col = [output_name]
        data = pd.concat([data, pd.DataFrame(prediction, columns = name_col)], axis = 1)
    if ratio == True:
        data['ratio_predictions'] = ratio_predictions(data[names_models[0]+'_predictions'], data[names_models[1]+'_predictions'],n)
    
    data  = pd.concat([inputs, data], axis = 1)
    return data
def compare_prediction_of_minimum(full_data:pd.DataFrame, output:str):
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
    rows_compared.to_csv('/Users/letiviada/dissertation_mmsc/regression/optimization/comparison.csv', index=False)
    return rows_compared

def make_plots(full_data:pd.DataFrame, output:str,actual: bool, prediction: bool,lines: bool,data_line:pd.DataFrame,type_data:str):
    opt_ml(full_data, output, actual, prediction,lines= lines,data_line=data_line,type_data = type_data)


if __name__ == '__main__':
    # Define the parameters
    filename = 'performance_indicators/performance_indicators_phi_1.0.json'
    time = 400
    n = 1
    outputs = ['volume_liquid', 'total_concentration']
    type_model = ['polynomial', 'gradient_boosting']

    # Get the data we have used to train the models
    full_data, vol_model, conc_model = get_full_data(outputs, time, n, type_model,filename)
    pd.set_option('display.max_columns', None)
    compare_prediction_of_minimum(full_data,'ratio')

    # Get the data for the plots
    alpha = np.arange(0.3,1.001,0.001)
    beta = np.arange(0.03, 0.1, 0.005).round(3)
    data_plots = get_data_for_opt([vol_model,conc_model],outputs, alpha, beta,ratio = True, n=n)
    print(data_plots.head())
    # Make the plots with the training data and the plotting predictions
   # make_plots(full_data,'volume_liquid_time_400', actual = True, prediction = True,lines = True, data_line = data_plots,type_data = 'large') 
    #make_plots(full_data,'total_concentration_time_400', actual = True, prediction = True,lines = False, data_line = data_plots,type_data = 'large')
    #make_plots(full_data,'ratio', actual = True, prediction = True,lines = False, data_line = data_plots, type_data = 'large')


