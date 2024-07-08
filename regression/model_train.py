from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, KFold
from utils import save_model
from utils import clean_data, obtain_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
def main():
    data_to_keep = clean_data(filename='performance_indicators_phi_1.0.json')
    data = obtain_data(data_to_keep, 'Termination time')
    inputs, outputs = data.drop('Termination time', axis = 1), data['Termination time']
   
    # Split the data into training and testing sets
    # ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Create the model
    # ----------------
    rand_forest_model = RandomForestRegressor(random_state=42)

    # Hyperparameter tuning
    # ---------------------
    param_grid = {
        'n_estimators': [200,300], # Number of trees in the forest
        'max_depth': [10,20], # Maximum depth of the tree
        # Minimum number of samples required to split an internal node 
        'ccp_alpha': np.arange(0, 1.1, 0.1), # Complexity parameter used for Minimal Cost-Complexity Pruning
    }
    grid_search = GridSearchCV(rand_forest_model, param_grid,
                    cv = 10,
                      n_jobs=-1, verbose=1,return_train_score=True)
    # n_jobs = -1 means use all processors, cv = 5 means 5-fold cross validation
    start = time.time()
    grid_search.fit(X_train, y_train)
    end = time.time()
    print("Best parameters:" , grid_search.best_estimator_)
    print(f'Best score:{grid_search.best_score_} ')
    #print(f'Time taken: {end-start}')
    model_regression = grid_search.best_estimator_
    #oob_score = model_regression.oob_score_
    #print(f'Out of bag score: {oob_score}')
    
    # Access the cv_results_ attribute
    cv_results = grid_search.cv_results_

    # Find the index of the best estimator
    best_index = grid_search.best_index_

    mean_test_score = cv_results['mean_test_score'][best_index]
    std_test_score = cv_results['std_test_score'][best_index]
    split_test_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

    # Extract the mean train score and the scores of each fold for the best estimator
    #mean_train_score = cv_results['mean_train_score'][best_index]
    #std_train_score = cv_results['std_train_score'][best_index]
    #split_train_scores = [cv_results[f'split{i}_train_score'][best_index] for i in range(10)]

    print(f"Mean validation score: {mean_test_score}")
    print(f"Standard deviation of validation score: {std_test_score}")
    print(f"Validation scores for each split: {split_test_scores}")

    #print(f"Mean train score: {mean_train_score}")
    #print(f"Standard deviation of train score: {std_train_score}")
    #print(f"Train scores for each split: {split_train_scores}")

    #model_regression.fit(X_train, y_train)

    # Predict the values
    # -------------------
    y_pred = model_regression.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2 score for test: {r2}")

    # Save the model
    # --------------
    #save_model(model_regression, 'regression/models/termination_time_model.pkl')
    #save_model((X_test, y_test), 'regression/models/test_data.pkl')
    #save_model((X_train, y_train), 'regression/models/train_data.pkl')

if __name__ == '__main__':
    main()
