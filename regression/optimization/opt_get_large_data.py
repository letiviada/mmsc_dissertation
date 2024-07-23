import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
import pandas as pd
import numpy as np
from utils import ratio_predictions

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
    for i, model in enumerate(models):
        prediction = model.predict(inputs)
        output_name = f'{names_models[i]}_predictions'
        name_col = [output_name]
        data = pd.concat([data, pd.DataFrame(prediction, columns = name_col)], axis = 1)
    if ratio == True:
        data['ratio_predictions'] = ratio_predictions(data[names_models[0]+'_predictions'], data[names_models[1]+'_predictions'],n)
    
    data  = pd.concat([inputs, data], axis = 1)
    return data
