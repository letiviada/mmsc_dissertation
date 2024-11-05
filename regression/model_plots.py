import matplotlib.pyplot as plt
import sys
# Append specific paths to system path for module imports (adapt as needed)
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression/')
from models.model_eval import model_eval
from optimization.opt_time.make_models import ml_model
from utils_r import clean_data, obtain_data, change_name_time , get_data_from_json
# Additional paths added to the system for further module imports
sys.path.append('/Users/letiviada/dissertation_mmsc/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
# Import custom functions from modules in the added paths
from multiscale.plotting import scatter_solutions, model_plot_with_lines_and_scatter
import numpy as np
import pandas as pd

# Define a function to generate data for plotting lines based on model predictions
def get_data_lines(input1, input2, output_variable, type_model):
    # Create a grid of input parameters using the meshgrid function and reshape for 2D use
    alpha_beta_grid = np.array(np.meshgrid(input1, input2)).T.reshape(-1, 2)
    inputs = pd.DataFrame(alpha_beta_grid, columns=['adhesivity', 'particle_size'])
    # Initialize an empty DataFrame to store model predictions
    data = pd.DataFrame()
    # Use the custom ml_model function to generate predictions based on inputs
    prediction, _ = ml_model(inputs,output_variable,time = None, type_model = type_model)
    # Store predictions in a DataFrame, merging with input values for clarity
    data =  pd.DataFrame(prediction, columns = [output_variable])
    data = pd.concat([inputs, data], axis = 1)
    return data

# Define a function to retrieve and clean large dataset from a file (large dataset is a fine grid of the input grid in the multiscale model)
def get_large_set_of_data(output_variable,filename):
    # Clean data using a custom clean_data function, which preprocesses the data
    physical_data = clean_data(filename)
    # Extract only the relevant data related to the specified output variable
    data_model = obtain_data([output_variable], physical_data)
    return data_model

# Define the main function to evaluate models and generate plots
def main(output_variable, name_eval, type_model,filename):  
    # Evaluate the model using custom model_eval function to get input and output data
    inputs, outputs = model_eval(output_variable,name_eval,type_model)
    # Define ranges for two model parameters, alpha and beta
    alpha = np.arange(0.2,1.001,0.0001) # Continuous range of values for alpha
    beta = np.arange(0.02, 0.04, 0.005).round(3)  # Discrete values for beta
    beta = [0.02, 0.025, 0.03, 0.035, 0.04]  # Overwrite beta with specific values
    # Get model prediction data for plotting lines
    data_lines = get_data_lines(alpha,beta,output_variable, type_model)
    # Retrieve a cleaned dataset for model comparison
    data_model = get_large_set_of_data(output_variable, filename)
    # Generate a plot with lines and scatter points for model evaluation
    model_plot_with_lines_and_scatter(inputs, outputs, output_variable, type_model, data_lines,data_model)

if __name__ == '__main__':
    names =['total'] # Names can be total, train or test (they give the plots of the data of ch4 with dots and lines)
    filename = 'performance_indicators/performance_indicators_phi_4.0.json'
    output_variables = ['termination_time', 'total_throughput','efficiency']

    for output_variable in output_variables:
        for name_eval in names:
            for type_model in ['polynomial','gradient_boosting', 'random_forest']:
                main(output_variable, name_eval,type_model,filename) 





