from sklearn.model_selection import train_test_split
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import clean_data, obtain_data, sampling_data, clean_data, save_data_to_csv, data_time
from models import train_model
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


def sample_size_model(output:str, run:int, size: int, filename:str, type_model:str, sampling_method:str):
    ml_data = clean_data(filename)
    data = obtain_data([output],ml_data)
    X_train, mse, r2 = train_model(output, data, size_train = size, run = run, type_model=type_model,size_sampling = sampling_method,save = False)
    return X_train, mse, r2


def sample_size_model_time(output:str,run: int, size:int, time:int, filename:str, type_model:str):
    ml_data = clean_data(filename)
    data = data_time(time, [output], ml_data)
    output_name = f'{output}_time_{time}'
    data_train = obtain_data([output_name], data)
    X_train, mse, r2 = train_model(output_name, data_train,size_train=size, run = run, type_model = type_model, save = False)
    return X_train, mse, r2

def run_model(output,num_runs,size,filename,type_model, time, sampling_method):
    mse_list = []
    r2_list = []
    point_counts = defaultdict(int)
    for run in range(num_runs):
        if time is None:
            X_train, mse, r2 = sample_size_model(output,run, size,filename,type_model,sampling_method)
        else:
            X_train, mse, r2 = sample_size_model_time(output,run, size,time,filename,type_model)

        mse_list.append(mse)
        r2_list.append(r2)
        for _, row in X_train.iterrows():
            point = tuple(row)
            point_counts[point] += 1


    results = pd.DataFrame({
        'MSE': mse_list,
        'R2': r2_list
    })
    
    mean_mse = results['MSE'].mean()
    std_mse = results['MSE'].std()
    mean_r2 = results['R2'].mean()
    std_r2 = results['R2'].std()
    point_counts_df = pd.DataFrame(list(point_counts.items()), columns=['Point', 'Count'])
    
    return results, mean_mse, std_mse, mean_r2, std_r2, point_counts_df

def main(output,num_runs,sizes,filename,type_model,sampling_method,time = None):
    mean_r2_list = []
    std_r2_list = []
    for size in sizes:
        print(f"Running model for sample size: {size}")
        _, mean_mse, std_mse, mean_r2, std_r2, point_counts_df = run_model(output,num_runs, size,filename,type_model, time, sampling_method)
        mean_r2_list.append(mean_r2)
        std_r2_list.append(std_r2)
    # Create a DataFrame to store the summary statistics
        save_data_to_csv(point_counts_df,f'sample_size_study/{output}/choose_points/{type_model}/{sampling_method}',f'point_counts_{size}_{type_model}_{sampling_method}.csv')
    summary_stats = pd.DataFrame({
        'Sample Size': sizes,
        'Mean R2': mean_r2_list,
        'Std R2': std_r2_list
    })
    save_data_to_csv(summary_stats,f'sample_size_study/{output}',f'summary_statistics_{type_model}_{sampling_method}.csv')
    print(f'Sample size study complete for {output} and {type_model}')

if __name__ == '__main__':
    filename = 'performance_indicators/performance_indicators_sample_size.json'
    num_runs = 2
    sizes = np.arange(180,181,1)
    sampling_methods = ['latin_hypercube']
    type_models = ['polynomial'] #['polynomial','random_forest','gradient_boosting']
    for output in ['termination_time']: #,'efficiency']:
        for type_model in type_models:
            for sampling_method in sampling_methods:
                main(output,num_runs, sizes,filename,type_model, sampling_method  = sampling_method)
