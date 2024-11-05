from model_train import train_model
from utils_r import get_data_from_json, data_time, clean_data
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression') 
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import obtain_data
import pandas as pd

def first_model(outputs:str, filename:str, type_model:str):
    # Forward model discussed in Ch4
    ml_data = clean_data(filename) 
    for output in outputs:
        data = obtain_data([output],ml_data)
        train_model(output, data,type_model=type_model)

def opt_model(time, outputs:str, filename:str, type_model:str):
    # This model is the one that considers the throughput/volume and efficiency at a specific time (Ch5.2)
    ml_data = get_data_from_json(filename) # Load the data from the JSON file that is from the multiscale model
    data = data_time(time, ml_data) # Get the model data at the specific time
    for output in outputs:
        output_name = f'{output}_time_{time}'
        data_train = obtain_data([output_name], data) # Clean the data to get the
        train_model(output_name, data_train, type_model = type_model)

def model_from_csv(filename:str, type_model:str): 
    # This model is the one that finds the efficiency and time to get to a throughput/volume of a certain value (Ch5.3)
    data = pd.read_csv(filename) # The data has been stored in a csv file (see folder optimization -> opt_throughput)
    for output in data.columns[2:-1]:
        data_train = obtain_data([output], data) # Clean the data to get the data train which has 3 columns namely: alpha, beta, output
        train_model(output, data_train, type_model = type_model) # Calls the function that trains the model and saves the trained model


if __name__ == '__main__':
    filename = 'performance_indicators/performance_indicators_opt.json' # The data obtained from the multiscale model (can be changed by other files)
    csv_file = 'regression/optimization/opt_throughput/data/throughput_100/initial_dataset.csv' 
    # csv_file contains the dataset that is obtained when considering the value of the outputs at a specific throughput -- we have cleaned the data so that values of pore adherence and particle size that give total throughput less than the set value are not includeed
    outputs = ['volume', 'efficiency'] # Outputs considered for the model of a specific time
    output_first = ['total_throughput', 'termination_time','efficiency'] # Outputs considered fort the forward model
    time = 400 # Set the time
    for type_model  in ['polynomial', 'random_forest','gradient_boosting']: # for loop to run the models for the different situations
        opt_model(time, outputs, filename, type_model) # Call model from Ch 5.2
        first_model(output_first, filename, type_model) # Call  model from Ch 4
        model_from_csv(csv_file, type_model) # Call  model from Ch 5.3
