from sklearn.model_selection import train_test_split, GridSearchCV
from utils import save_model
from utils import obtain_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
def main():
    data = obtain_data('termination_time')
    inputs, ouputs = data[['alpha', 'beta']], data['termination_time']
    start = time.time()
    # Split the data into training and testing sets
    # ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(inputs, ouputs, test_size=0.2, random_state=42)

    # Create the model
    # ----------------
    rand_forest_model = RandomForestRegressor(random_state=42)

    # Hyperparameter tuning
    # ---------------------
    param_grid = {
        'n_estimators': [100,200,300], # Number of trees in the forest
        'max_depth': [10, 20, 30], # Maximum depth of the tree
        'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
        'max_features': [1,2], # Number of features to consider when looking for the best split (I think because my features
        # are alpha and beta, I should only consider 1 or 2 features at a time)
        'ccp_alpha': np.arange(0, 1.1, 0.1) # Complexity parameter used for Minimal Cost-Complexity Pruning
    }
    grid_search = GridSearchCV(rand_forest_model, param_grid, cv = 5, n_jobs= -1) 
    # n_jobs = -1 means use all processors, cv = 5 means 5-fold cross validation
    grid_search.fit(X_train, y_train)
    print("Best parameters:" , grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Train the model
    # ---------------
    best_model.fit(X_train, y_train)
    end = time.time()
    print(f"Time taken: {end-start} seconds")

    # Predict the values
    # -------------------
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2_score = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2 score: {r2_score}")

    # Save the model
    # --------------
    save_model(best_model, 'regression/models/termination_time_model.pkl')
    save_model((X_test, y_test), 'regression/models/test_data.pkl')
    save_model((X_train, y_train), 'regression/models/train_data.pkl')

if __name__ == '__main__':
    main()