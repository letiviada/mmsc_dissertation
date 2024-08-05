import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression/')
from models.model_eval import model_eval
from optimization.opt_time.make_models import ml_model
from utils_r import clean_data, obtain_data, change_name_time 
sys.path.append('/Users/letiviada/dissertation_mmsc/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
from multiscale.plotting import scatter_solutions, model_plot_with_lines_and_scatter
import numpy as np
import pandas as pd

def get_data_lines(input1, input2, output_variable, type_model):
    alpha_beta_grid = np.array(np.meshgrid(input1, input2)).T.reshape(-1, 2)
    inputs = pd.DataFrame(alpha_beta_grid, columns=['adhesivity', 'particle_size'])
    data = pd.DataFrame()
    prediction, _ = ml_model(inputs,output_variable,time = None, type_model = type_model)
    data =  pd.DataFrame(prediction, columns = [output_variable])
    data = pd.concat([inputs, data], axis = 1)
    return data

def get_large_set_of_data(output_variable,filename):
    physical_data = clean_data(filename)
    data_model = obtain_data([output_variable], physical_data)
    return data_model

def main(output_variable, name_eval, type_model,filename):  
    inputs, outputs = model_eval(output_variable,name_eval,type_model)
    #scatter_solutions(inputs,outputs,output_variable, type_model,name_eval)
    alpha = np.arange(0.3,1.001,0.0001)
    beta = np.arange(0.03,0.095,0.01).round(2)
    data_lines = get_data_lines(alpha,beta, output_variable, type_model)
    data_model = get_large_set_of_data(output_variable, filename)
    model_plot_with_lines_and_scatter(inputs, outputs, output_variable, type_model, data_lines,data_model)

if __name__ == '__main__':
    names =['train']
    filename = 'performance_indicators/performance_indicators_standard_basic.json'
    #output_variable = 'volume_liquid_time_400'
    output_variables = ['termination_time', 'total_throughput','efficiency']

    for output_variable in output_variables:
        for name_eval in names:
            for type_model in ['polynomial']: #, 'random_forest']:
                main(output_variable, name_eval,type_model,filename) 






