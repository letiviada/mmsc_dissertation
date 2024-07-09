from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, KFold
from utils import save_model
from utils import clean_data, obtain_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
def main(output):
    data_to_keep = clean_data(filename='performance_indicators_phi_1.0.json')
    data = obtain_data(data_to_keep, output)
    inputs, outputs = data.drop(output, axis = 1), data[output]
   
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
        'max_depth': [5,10], # Maximum depth of the tree
        'ccp_alpha': np.arange(0, 1.1, 0.1), # Complexity parameter used for Minimal Cost-Complexity Pruning
    }
    grid_search = GridSearchCV(rand_forest_model, param_grid,
                    cv = 8, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
    model_regression = grid_search.best_estimator_
    #oob_score = model_regression.oob_score_
    #print(f'Out of bag score: {oob_score}')
    
    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_
    mean_test_score = cv_results['mean_test_score'][best_index]
    std_test_score = cv_results['std_test_score'][best_index]
    print(f"Mean score: {mean_test_score}")
    print(f"Standard deviation score: {std_test_score}")

    # Predict the values
    # -------------------
    y_pred = model_regression.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2 score for test: {r2}")

    # Save the model
    # --------------
    save_model(model_regression, f'regression/models/{output}/model.pkl')
    save_model((X_test, y_test), f'regression/models/{output}/test_data.pkl')
    save_model((X_train, y_train), f'regression/models/{output}/train_data.pkl')

if __name__ == '__main__':
    main('Termination time')
   # main('Lifetime')
