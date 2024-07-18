import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from opt_get_large_data import get_data_for_opt
from opt_get_models import get_full_data_and_models, get_physical_model
from multiscale.plotting import opt_ml, make_loglog
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_plots(full_data:pd.DataFrame, output:str,actual: bool, prediction: bool,lines: bool, data_line:pd.DataFrame,type_data:str,particle_sizes: list= None):
    opt_ml(full_data, output, actual, prediction,lines= lines,data_line=data_line,type_data = type_data, particle_sizes= particle_sizes)

def get_data_after_ml(alpha, beta, outputs, time, n, type_model, filename):
    full_data, vol_model, conc_model = get_full_data_and_models(outputs, time, n, type_model,filename)
    #alpha = np.arange(0.3,1.001,0.001)
    #beta = np.arange(0.03, 0.1, 0.005).round(3)
    data_plots = get_data_for_opt([vol_model,conc_model],outputs, alpha, beta,ratio = True, n=n)
    #make_plots(full_data,'volume_liquid_time_400', actual = True, prediction = True,lines = True, data_line = data_plots,type_data = 'large') 
    #make_plots(full_data,'total_concentration_time_400', actual = True, prediction = True,lines = False, data_line = data_plots,type_data = 'large')
    make_plots(full_data,'ratio', actual = True, prediction = True,lines = False, data_line = data_plots, type_data = 'large')
    return data_plots
def get_data_varyiing_n(particle_size):
    
    pass

if __name__ == '__main__':
    
    # Define the parameters
    filename = 'performance_indicators/performance_indicators_standard.json'
    time = 400
    n = 1
    particle_sizes = [0.03,0.04,.05,0.06,0.07,0.08,0000.09]
    data_ratio = get_physical_model(['volume_liquid','total_concentration'],time,n,filename)
    #make_plots(data_ratio,'volume_liquid_time_400', actual = True, prediction = False,lines = True, data_line = None,type_data = 'standard') 
   # make_plots(data_ratio,'total_concentration_time_400', actual = True, prediction = False,lines = False, data_line = None,type_data = 'standard',particle_sizes= particle_sizes)
    make_loglog(data_ratio,'volume_liquid_time_400', betas = particle_sizes,type_data='standard')
    plt.show()


