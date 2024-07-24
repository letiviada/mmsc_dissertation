
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from models.model_train import  train_model
from opt_get_models import  get_data_opt_adhesivity_plots
sys.path.append('/Users/letiviada/dissertation_mmsc/multiscale/')
sys.path.append('/home/viadacampos/Documents/dissertation_mmsc/multiscale/')
from multiscale.plotting import plot_optimum
import pandas as pd

def train_and_plot_opt_adhesivity(train:bool, plot: bool, type_model: str):
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
    data = pd.read_csv('regression/optimization/optimum_values.csv')
    output = 'adhesivity'

    if train ==True:
        train_model(output, data, size_train = 'all', type_model = type_model, save = True)

    if plot == True:
        data_pred, model = get_data_opt_adhesivity_plots(data, type_model = type_model)
        # Plot the data
        plot_optimum(data_pred,particle_sizes = 'all', actual= True, predictions = True, save = True)

if __name__ == '__main__':
    train_and_plot_opt_adhesivity(train = False, plot = True, type_model = 'gradient_boosting')

