import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from regression.optimization.opt_time.old.opt_get_large_data import get_data_for_opt
from regression.optimization.opt_time.make_models import get_full_data_and_models, get_physical_model_time
from multiscale.plotting import opt_ml, make_loglog, plot_optimum
import pandas as pd
import numpy as np

def make_plots(full_data:pd.DataFrame, output:str,actual: bool, prediction: bool,lines: bool,
                data_line:pd.DataFrame,type_data:str,particle_sizes: list= None, save: bool = True):
    opt_ml(full_data, output, actual, prediction,lines = lines,
           data_line=data_line,type_data = type_data, particle_sizes = particle_sizes, save = save)

def get_data_after_ml(alpha, beta, outputs, time, n, type_model, filename):
    full_data, vol_model, conc_model = get_full_data_and_models(outputs, time, n, type_model,filename)
    #alpha = np.arange(0.3,1.001,0.001)
    #beta = np.arange(0.03, 0.1, 0.005).round(3)
    data_plots = get_data_for_opt([vol_model,conc_model],outputs, alpha, beta,ratio = True, n=n)
    #make_plots(full_data,'volume_liquid_time_400', actual = True, prediction = True,lines = True, data_line = data_plots,type_data = 'large') 
    #make_plots(full_data,'total_concentration_time_400', actual = True, prediction = True,lines = False, data_line = data_plots,type_data = 'large')
    make_plots(full_data,'ratio', actual = True, prediction = True,lines = False, data_line = data_plots, type_data = 'large')
    return data_plots


if __name__ == '__main__':
    
  # Define the parameters
    filename = 'performance_indicators/performance_indicators_standard_basic.json'
    time = 400
    n_values = np.arange(0.04,3.25,0.05).round(2)
    particle_sizes = 'all'
    data_ratio = get_physical_model_time(['volume_liquid','total_concentration'],time,1,filename)

    vol_name = f'volume_liquid_time_{time}'
    conc_name = f'total_concentration_time_{time}'
    make_plots(data_ratio,vol_name, actual = True, prediction = False,lines = True, data_line = None,type_data = 'standard',  particle_sizes= particle_sizes, save= False) 
    make_plots(data_ratio,conc_name, actual = True,  prediction = False,lines = False, data_line = None,type_data = 'standard',particle_sizes= particle_sizes, save= False)
    make_loglog(data_ratio,f'volume_liquid_time_{time}', betas = particle_sizes,type_data='standard')

    #data = pd.read_csv(f'regression/optimization/opt_time/data/time_{time}/optimum_values.csv')
    #plot_optimum(data,particle_sizes,predictions = False)

 

