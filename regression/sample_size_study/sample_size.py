from sklearn.model_selection import train_test_split
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import clean_data, obtain_data, sampling_data, get_data_from_json, save_data_to_csv, data_time
from models import train_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


def sample_size_model(output:str, size: int, filename:str, type_model:str, sampling_method:str):
    ml_data = get_data_from_json(filename)
    data = obtain_data([output],ml_data)
   # data = clean_data(data_b)
    mse, r2 = train_model(output, data, size_train = size, type_model=type_model,size_sampling = sampling_method,save = False)
    return mse, r2


def sample_size_model_time(output:str, size:int, time:int, filename:str, type_model:str):
    ml_data = get_data_from_json(filename)
    data = data_time(time, [output], ml_data)
    output_name = f'{output}_time_{time}'
    data_train = obtain_data([output_name], data)
    mse, r2 = train_model(output_name, data_train,size_train=size, type_model = type_model, save = False)
    return mse, r2

def run_model(output,num_runs,size,filename,type_model, time, sampling_method):
    mse_list = []
    r2_list = []
    for _ in range(num_runs):
        if time is None:
            mse, r2 = sample_size_model(output,size,filename,type_model,sampling_method)
        else:
            mse, r2 = sample_size_model_time(output,size,time,filename,type_model)

        mse_list.append(mse)
        r2_list.append(r2)

    results = pd.DataFrame({
        'MSE': mse_list,
        'R2': r2_list
    })
    
    mean_mse = results['MSE'].mean()
    std_mse = results['MSE'].std()
    mean_r2 = results['R2'].mean()
    std_r2 = results['R2'].std()
    
    return results, mean_mse, std_mse, mean_r2, std_r2

def main(output,num_runs,sizes,filename,type_model,sampling_method,time = None):
    mean_r2_list = []
    std_r2_list = []
    for size in sizes:
        print(f"Running model for sample size: {size}")
        _, mean_mse, std_mse, mean_r2, std_r2 = run_model(output,num_runs, size,filename,type_model, time, sampling_method)
        mean_r2_list.append(mean_r2)
        std_r2_list.append(std_r2)
    # Create a DataFrame to store the summary statistics
    summary_stats = pd.DataFrame({
        'Sample Size': sizes,
        'Mean R2': mean_r2_list,
        'Std R2': std_r2_list
    })
    save_data_to_csv(summary_stats,f'sample_size_study/{output}',f'summary_statistics_{type_model}_{sampling_method}.csv')
   # print(f'Sample size study complete for {output} and {type_model}')

if __name__ == '__main__':
    filename = 'performance_indicators/performance_indicators_sample_size.json'
    num_runs = 10
    sizes = np.arange(31,181,1)
    sampling_methods = ['latin_hypercube']
    type_models = ['gradient_boosting'] #['polynomial','random_forest','gradient_boosting']
    for output in ['termination_time']: #,'efficiency']:
        for type_model in type_models:
            for sampling_method in sampling_methods:
                main(output,num_runs, sizes,filename,type_model, sampling_method  = sampling_method)
