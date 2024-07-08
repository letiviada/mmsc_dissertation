from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, KFold
from utils import save_model
from utils import clean_data, obtain_data, sampling_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import time
import numpy as np


def model(size):
    data_to_keep = clean_data(filename='performance_indicators_phi_1.0.json')
    data = obtain_data(data_to_keep, 'Termination time')
    inputs, outputs = data.drop('Termination time', axis = 1), data['Termination time']
   # print(inputs.shape, outputs.shape)
    X_train_old, X_test, y_train_old, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    # Resample data
    X_train, y_train = sampling_data(X_train_old, y_train_old, size)
    # Create the model
    random_forest = RandomForestRegressor(n_estimators = 200, max_depth = 20, ccp_alpha = 0.9, random_state = 42)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def run_model(num_runs,size):
    mse_list = []
    r2_list = []
    for _ in range(num_runs):
        mse, r2 = model(size)
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

def main(num_runs,sizes):
    mean_r2_list = []
    std_r2_list = []
    for size in sizes:
        print(f"Running model for sample size: {size}")
        _, mean_mse, std_mse, mean_r2, std_r2 = run_model(num_runs, size)
        mean_r2_list.append(mean_r2)
        std_r2_list.append(std_r2)
    # Create a DataFrame to store the summary statistics
    summary_stats = pd.DataFrame({
        'Sample Size': sizes,
        'Mean R2': mean_r2_list,
        'Std R2': std_r2_list
    })
    summary_stats.to_csv('/home/viadacampos/Documents/mmsc_dissertation/summary_statistics_by_sample_size.csv', index=False)

if __name__ == '__main__':
    num_runs = 10
    sizes = np.arange(10,283,2)
    main(num_runs, sizes)

    
    
