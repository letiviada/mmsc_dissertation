from sklearn.model_selection import train_test_split, GridSearchCV
from utils import clean_data, obtain_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import scatter_solutions

def model_eval(output,type_eval):

    file = "regression/models/model_1_alpha_large_betta_small/termination_time_model.pkl"
    best_model = joblib.load(file)

    if type_eval == 'total':
        data_to_keep = clean_data()
        data = obtain_data(data_to_keep, output)
        inputs, outputs = data.drop(output, axis = 1), data[output]
    else:
        inputs, outputs = joblib.load(f"regression/models/model_1_alpha_large_betta_small/{type_eval}_data.pkl")
    

    y_pred = best_model.predict(inputs)
    outputs = outputs.to_frame()
    outputs['Prediction'] = y_pred.tolist()
    mse = mean_squared_error(outputs[output], outputs['Prediction']) 
    r2 = r2_score(outputs[output], outputs['Prediction'])
    print(f"Mean squared error, {type_eval} : {mse}")
    print(f"R2 score,{type_eval} : {r2}")
    return inputs, outputs
    
def main(type_eval):  
    inputs, outputs = model_eval(type_eval)
    scatter_solutions(inputs,outputs,type_eval)

if __name__ == '__main__':
    names =['total', 'train', 'test']
    output = 'Termination time'
    for name_eval in names:
        main(output, name_eval) 





