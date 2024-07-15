import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import open_model, change_name_time, open_data_model
from utils import get_data_from_json, data_time, obtain_data, get_ratio
import matplotlib.pyplot as plt
import numpy as np

def get_models(output,time, type_model):
    output_name = change_name_time(output, time)
    model_path = f'regression/models_{type_model}/'
    model = open_model(output = output_name, model_path = model_path)
    inputs, outputs = open_data_model('total', output_name, model_path = model_path)
    predictions = model.predict(inputs)
    return predictions, outputs

def get_physical_model():
    # Physical Model
    physical_data = get_data_from_json('performance_indicators/performance_indicators_standard.json')
    inputs, outputs = open_data_model('total', 'total_concentration_time_400', model_path = f'regression/models_polynomial/')
    data = data_time(400, ['volume_liquid', 'total_concentration'], physical_data)
    data_model = obtain_data(['volume_liquid_time_400', 'total_concentration_time_400'], data)
    data_ratio = get_ratio('volume_liquid_time_400', 'total_concentration_time_400',n, data_model)

output_name = 'volume_liquid_time_400'
time = 400
n = 1

def main(n):
    volume_predictions = get_ml_model('volume_liquid_time_400', 'polynomial')
    concentration_predictions = get_ml_model('total_concentration_time_400', 'polynomial')
    ratio = (volume_predictions ** n)/(concentration_predictions)


