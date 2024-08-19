from model_train import train_model
from utils_r import get_data_from_json, data_time, clean_data
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import obtain_data
import pandas as pd

def first_model(outputs:str, filename:str, type_model:str):
    ml_data = clean_data(filename)
    for output in outputs:
        data = obtain_data([output],ml_data)
        train_model(output, data,type_model=type_model)

def opt_model(time, outputs:str, filename:str, type_model:str):
    ml_data = get_data_from_json(filename)
    data = data_time(time, ml_data)
    for output in outputs:
        output_name = f'{output}_time_{time}'
        data_train = obtain_data([output_name], data)
        train_model(output_name, data_train, type_model = type_model)
def model_from_csv(filename:str, type_model:str):
    data = pd.read_csv(filename)
    for output in data.columns[2:-1]:
        data_train = obtain_data([output], data)
        train_model(output, data_train, type_model = type_model)


if __name__ == '__main__':
    filename = 'performance_indicators/performance_indicators_opt.json'
    csv_file = 'regression/optimization/opt_throughput/data/throughput_100/initial_dataset.csv'
    outputs = ['volume','efficiency']
    output_first = ['total_throughput', 'termination_time','efficiency']
    time = 400
    #first_model(output_first, filename,'polynomial')
    for type_model  in ['polynomial']: # 'random_forest','gradient_boosting']:
        opt_model(time, outputs, filename, type_model)
        #first_model(output_first, filename, type_model)
       # model_from_csv(csv_file, type_model)
