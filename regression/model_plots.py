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

def model_eval(type_eval):

    file = "regression/models/termination_time_model.pkl"
    best_model = joblib.load(file)

    if type_eval == 'total':
        data_to_keep = clean_data()
        data = obtain_data(data_to_keep, 'Termination time')
        inputs, outputs = data.drop('Termination time', axis = 1), data['Termination time']
    else:
        inputs, outputs = joblib.load(f"regression/models/{type_eval}_data.pkl")
    

    y_pred = best_model.predict(inputs)
    outputs = outputs.to_frame()
    outputs['Prediction'] = y_pred.tolist()
    mse = mean_squared_error(outputs['Termination time'], outputs['Prediction']) 
    r2 = r2_score(outputs['Termination time'], outputs['Prediction'])
    print(f"Mean squared error, {type_eval} : {mse}")
    print(f"R2 score,{type_eval} : {r2}")
    return inputs, outputs
    
def main(type_eval):  
    inputs, outputs = model_eval(type_eval)
    scatter_solutions(inputs,outputs,type_eval)

if __name__ == '__main__':
    names =['total', 'train', 'test']
    for name_eval in names:
        main(name_eval) 





