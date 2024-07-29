
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression/')
from utils import open_model
from models.model_train import  train_model
sys.path.append('/Users/letiviada/dissertation_mmsc/multiscale/')
sys.path.append('/home/viadacampos/Documents/dissertation_mmsc/multiscale/')
from multiscale.plotting import plot_optimum
import pandas as pd

def get_data_opt_adhesivity_plots(data,name_model,type_model = 'random_forest')->tuple:
    """
    Function that gets the data for the adhesivity plots

    Parameters:
    -----------
    data (pd.DataFrame): the data to be considered

    Returns:
    --------
    data_plots (pd.DataFrame): the data for the plots
    """
    model_path = f'regression/models_{type_model}/'
    ml_model = open_model(f'adhesivity_{name_model}',model_path=model_path)
    inputs = data.drop('adhesivity', axis=1)
    predictions = ml_model.predict(inputs)
    data['adhesivity_predictions'] = predictions
    return  data, ml_model

def train_and_plot_opt_adhesivity(model: str,value:int, train:bool, plot: bool, type_model: str):
    """
    Function that trains the model and plots the data

    Parameters:
    -----------
    train (bool): whether to train the model or not
    plot (bool): whether to plot the data or not
    type_model (str): the type of model to be used (polynomial, random_forest, gradient_boosting)

    Returns:
    --------
    None
    
    """
    data = pd.read_csv(f'regression/optimization/opt_{model}/data/{model}_{value}/optimum_values.csv')
    output = 'adhesivity'

    if train ==True:
        train_model(output, data, size_train = 'all', type_model = type_model, save = True)

    if plot == True:
        data_pred, model = get_data_opt_adhesivity_plots(data, name_model = f'{model}_{value}',type_model = type_model)
        # Plot the data
        plot_optimum(data_pred,particle_sizes ='all', actual= True, predictions = False, save = True)

if __name__ == '__main__':
   # train_and_plot_opt_adhesivity(model = 'throughput', value = 100, train = False, plot = False, type_model = 'gradient_boosting')
    train_and_plot_opt_adhesivity(model = 'time', value = 400, train = False, plot = False, type_model = 'gradient_boosting')
