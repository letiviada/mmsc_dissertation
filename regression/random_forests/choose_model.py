from model_train import train_model
from utils import get_data_from_json, data_time
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import obtain_data


def first_model(outputs:str, filename:str):
    ml_data = get_data_from_json(filename)
    for output in outputs:
        data = obtain_data([output],ml_data)
        train_model(output, data)


def opt_model(time, outputs:str, filename:str, type_model:str):
    ml_data = get_data_from_json(filename)
    data = data_time(time, outputs, ml_data)
    for output in outputs:
        output_name = f'{output}_time_{time}'
        data_train = obtain_data([output_name], data)
        train_model(output_name, data_train, type_model)

if __name__ == '__main__':
    filename = 'performance_indicators/performance_indicators_phi_1.0.json'
    outputs =['volume_liquid', 'last_concentration']
    output_first = ['lifetime', 'termination_time','efficiency']
    time = 400
    opt_model(time, outputs, filename, 'gradient_boosting')
    #first_model(output_first, filename)
