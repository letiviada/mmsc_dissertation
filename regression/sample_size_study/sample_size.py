# Import necessary libraries
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Append custom paths to system path for importing custom modules
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')

# Import custom functions from utilities and models
from utils_r import clean_data, obtain_data, sampling_data, save_data_to_csv, data_time
from models import train_model

# Define function to train model for a specific sample size
def sample_size_model(output: str, run: int, size: int, filename: str, type_model: str, sampling_method: str):
    # Load and prepare data
    ml_data = clean_data(filename)
    data = obtain_data([output], ml_data)
    
    # Train model and get metrics
    X_train, mse, r2 = train_model(
        output, data, size_train=size, run=run, 
        type_model=type_model, size_sampling=sampling_method, save=False
    )
    return X_train, mse, r2

# Define function to train model for a specific sample size and time constraint
def sample_size_model_time(output: str, run: int, size: int, time: int, filename: str, type_model: str):
    # Load and filter data by time
    ml_data = clean_data(filename)
    data = data_time(time, [output], ml_data)
    
    # Define a new output name based on time and prepare data
    output_name = f'{output}time{time}'
    data_train = obtain_data([output_name], data)
    
    # Train model and get metrics
    X_train, mse, r2 = train_model(
        output_name, data_train, size_train=size, run=run, 
        type_model=type_model, save=False
    )
    return X_train, mse, r2

# Run the model multiple times and aggregate results
def run_model(output, num_runs, size, filename, type_model, time, sampling_method):
    mse_list = []
    r2_list = []
    point_counts = defaultdict(int)
    
    # Run model for specified number of iterations
    for run in range(num_runs):
        # Select appropriate function based on presence of time constraint
        if time is None:
            X_train, mse, r2 = sample_size_model(output, run, size, filename, type_model, sampling_method)
        else:
            X_train, mse, r2 = sample_size_model_time(output, run, size, time, filename, type_model)
        
        # Collect metrics and track occurrences of data points
        mse_list.append(mse)
        r2_list.append(r2)
        for _, row in X_train.iterrows():
            point = tuple(row)
            point_counts[point] += 1

    # Store results and compute summary statistics
    results = pd.DataFrame({'MSE': mse_list, 'R2': r2_list})
    mean_mse = results['MSE'].mean()
    std_mse = results['MSE'].std()
    mean_r2 = results['R2'].mean()
    std_r2 = results['R2'].std()
    point_counts_df = pd.DataFrame(list(point_counts.items()), columns=['Point', 'Count'])
    
    return results, mean_mse, std_mse, mean_r2, std_r2, point_counts_df

# Main function to execute sample size study and save results
def main(output, num_runs, sizes, filename, type_model, sampling_method, time=None):
    mean_r2_list = []
    std_r2_list = []
    
    # Loop through each sample size and run the model
    for size in sizes:
        print(f"Running model for sample size: {size}")
        _, mean_mse, std_mse, mean_r2, std_r2, point_counts_df = run_model(
            output, num_runs, size, filename, type_model, time, sampling_method
        )
        mean_r2_list.append(mean_r2)
        std_r2_list.append(std_r2)
        
        # Save data about point occurrences
        save_data_to_csv(
            point_counts_df, 
            f'sample_size_study/{output}/choose_points/{type_model}/{sampling_method}', 
            f'point_counts_{size}{type_model}{sampling_method}.csv'
        )
    
    # Store and save summary statistics
    summary_stats = pd.DataFrame({
        'Sample Size': sizes,
        'Mean R2': mean_r2_list,
        'Std R2': std_r2_list
    })
    save_data_to_csv(
        summary_stats, 
        f'sample_size_study/{output}', 
        f'summary_statistics_{type_model}_{sampling_method}.csv'
    )
    print(f'Sample size study complete for {output} and {type_model}')

# Execute script if run directly
if _name_ == '_main_':
    # Define parameters for model runs
    filename = 'performance_indicators/performance_indicators_sample_size.json'
    num_runs = 10
    sizes = np.arange(40, 181, 1)
    sampling_methods = ['random', 'latin_hypercube']
    type_models = ['gradient_boosting']  # Add more model types as needed
    
    # Specify outputs and loop through model types and sampling methods
    for output in ['termination_time']:  # Add more outputs as needed
        for type_model in type_models:
            for sampling_method in sampling_methods:
                main(
                    output, num_runs, sizes, filename, 
                    type_model, sampling_method=sampling_method
                )