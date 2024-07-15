from sklearn.model_selection import train_test_split, GridSearchCV
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import save_model, obtain_data, sampling_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
def train_model(output,data, size_train = 'all',type_model = 'random_forest'):
    # Obtain the data
    # ---------------
    inputs, outputs = data.drop(output, axis = 1), data[output]
    # Split the data into training and testing sets
    # ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    if size_train != 'all':
        X_train, y_train = sampling_data(X_train, y_train, size_train)
    # Create the model
    # ----------------
    if type_model == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
    elif type_model == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    # Hyperparameter tuning
    # ---------------------
    param_grid = {
        'n_estimators': [150,200,300], # Number of trees in the forest
        'max_depth': [10,20,30], # Maximum depth of the tree
        'ccp_alpha': np.arange(0, 1.1, 0.1), # Complexity parameter used for Minimal Cost-Complexity Pruning
    }
    grid_search = GridSearchCV(model, param_grid,
                    cv = 10, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
    model_regression = grid_search.best_estimator_
    # Predict the values
    # -------------------
    y_pred = model_regression.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2 score for test: {r2}")
    # Save the model
    # --------------
    save_model(model_regression, f'regression/models/{output}/model_{output}.pkl')
    save_model((X_test, y_test), f'regression/models/{output}/test_data.pkl')
    save_model((X_train, y_train), f'regression/models/{output}/train_data.pkl')
    save_model((inputs, outputs), f'regression/models/{output}/total_data.pkl')