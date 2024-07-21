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
    filename = 'performance_indicators/performance_indicators_standard.json'
    time = 400
    n_values = np.arange(0.04,3.25,0.01).round(2)
    particle_sizes = [0.08]
    data_ratio = get_physical_model(['volume_liquid','total_concentration'],time,1,filename)

    vol_name = f'volume_liquid_time_{time}'
    conc_name = f'total_concentration_time_{time}'
    #make_plots(data_ratio,vol_name, actual = True, prediction = False,lines = True, data_line = None,type_data = 'standard',  particle_sizes= particle_sizes, save= False) 
   # make_plots(data_ratio,conc_name, actual = True,  prediction = False,lines = False, data_line = None,type_data = 'standard',particle_sizes= particle_sizes, save= False)
#make_loglog(data_ratio,'volume_liquid_time_400', betas = particle_sizes,type_data='standard')

    data = pd.read_csv('regression/optimization/data_for_n_beta_0.03.csv')

    
    
    optimal_adhesivity = []
    pessimal_filter = []
  
    data = data.sort_values(by=f'total_concentration_time_{time}', ascending=True)
    filtered_data = data[(data[f'total_concentration_time_{time}'] <= 1)]
    filtered_data = filtered_data[(filtered_data[f'volume_liquid_time_{time}'] > 40)]
    adhesivity_keys = filtered_data['adhesivity'].unique()  
    for column in data.columns[4:]:
            max_value = filtered_data[column].max()
            min_value = filtered_data[column].min()
            optimal_adhesivity.append(filtered_data[filtered_data[column] == max_value]['adhesivity'].values[0])
            pessimal_filter.append(filtered_data[filtered_data[column] == min_value]['adhesivity'].values[0])
    
    plt.plot(n_values, optimal_adhesivity, label='Optimal')
    plt.plot(n_values, pessimal_filter, label = 'Pessimal')
    plt.xlabel('n')
    plt.ylabel('Adhesivity')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    #plt.xlabel('Adhesivity')
    #plt.ylabel('Value')
   # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   # plt.tight_layout()
   # plt.show()




