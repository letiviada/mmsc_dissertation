import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from opt_get_large_data import get_data_for_opt
from opt_get_models import get_full_data_and_models
from multiscale.plotting import opt_ml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    full_data, vol_model, conc_model = get_full_data_and_models(outputs, time, n, type_model,filename)

    # Get the data for the plots
    alpha = np.arange(0.3,1.001,0.001)
    beta = np.arange(0.03, 0.1, 0.005).round(3)
    data_plots = get_data_for_opt([vol_model,conc_model],outputs, alpha, beta,ratio = True, n=n)


    # Make the plots with the training data and the plotting predictions
    make_plots(full_data,'volume_liquid_time_400', actual = True, prediction = True,lines = True, data_line = data_plots,type_data = 'large') 
    make_plots(full_data,'total_concentration_time_400', actual = True, prediction = True,lines = False, data_line = data_plots,type_data = 'large')
    make_plots(full_data,'ratio', actual = True, prediction = True,lines = False, data_line = data_plots, type_data = 'large')
