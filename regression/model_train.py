from sklearn.model_selection import train_test_split, GridSearchCV
from treat_data import get_data_from_json, get_input_output
from sklearn.ensemble import RandomForestRegressor
import time
import joblib
import numpy as np

def main():
    data = get_input_output(get_data_from_json(filename = 'multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json'), 'termination_time')
    data = data.drop(data[(data['alpha'] == 0.3) & (data['beta'] == 0.03)].index)
    inputs = data[['alpha', 'beta']]
    ouputs = data['termination_time']

    start = time.time()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, ouputs, test_size=0.2, random_state=42)

    # Create the model
    rand_forest_model = RandomForestRegressor(random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100,200,300],
        'max_depth': [10, 20, 30],
        'ccp_alpha': np.arange(0, 1.0, 0.1)
    }
    grid_search = GridSearchCV(rand_forest_model, param_grid, cv = 5, n_jobs= -1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:" , grid_search.best_params_)
    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)
    
    joblib.dump(best_model, 'regression/models/termination_time_model.pkl')
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()