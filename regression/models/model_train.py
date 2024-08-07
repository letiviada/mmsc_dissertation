from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils_r import save_model, obtain_data, sampling_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def ensure_alpha_inclusion(data:pd.DataFrame, beta_col:str, alpha_col:str):
    """
    Function to ensure that the training dataset has at least one row for each beta value with the minimum and maximum alpha values.

    Parameters:
    -----------
    data (pd.DataFrame): the dataset
    beta_col (str): the column with the beta values
    alpha_col (str): the column with the alpha values

    Returns:
    --------
    edhe_rows (pd.DataFrame): the rows with the minimum and maximum alpha values for each beta value
    remaining_data (pd.DataFrame): the dataset without the mandatory rows
    """
    
    unique_betas = data[beta_col].unique() 
    edge_rows = pd.DataFrame()
    indices_to_drop = []
    
    for beta in unique_betas:
        beta_data = data[data[beta_col] == beta]
        min_alpha_index = beta_data[alpha_col].idxmin()
        max_alpha_index = beta_data[alpha_col].idxmax()
        min_alpha_row = beta_data.loc[min_alpha_index]
        max_alpha_row = beta_data.loc[max_alpha_index]
        edge_rows = pd.concat([edge_rows, min_alpha_row.to_frame().T, max_alpha_row.to_frame().T], ignore_index=True)
        indices_to_drop.extend([min_alpha_index, max_alpha_index])  # Add indices to the list

    # Drop rows from the original dataframe using the collected indices
    remaining_data = data.drop(indices_to_drop)

    return edge_rows, remaining_data

def train_model(output:str,data:pd.DataFrame, size_train = 'all',run = 0, size_sampling:str = 'random',type_model:str = 'random_forest',save:bool = True):
    """
    Function that trains a model for given output, dataset and type of model. Tunes hyperparameters using GridSearchCV, 
    fits and predicts the model for the best score and saves the model, the full dataset, the training and test data.

    Parameters:
    -----------
    output (str): the output to be predicted
    data (pd.DataFrame): the dataset
    size_train (int): the size of the training set
    type_model (str): the type of model to be used
    save (bool): whether to save the model or not, default is True

    Returns:
    --------
    None
    """
    inputs, outputs = data.drop(output, axis = 1), data[output]
    if type_model == 'polynomial':
        edges, inside = ensure_alpha_inclusion(data, 'particle_size', 'adhesivity')
        inputs_inside, outputs_inside = inside.drop(output, axis = 1), inside[output]
        inputs_edges, outputs_edges = edges.drop(output, axis = 1), edges[output]

    # Check if the training set is the whole dataset
        if size_train != 'all':
            left_size_train =  size_train - inputs_edges.shape[0]
            if size_train < inputs_edges.shape[0]:
                raise ValueError('The size of the points in the edges is larger than the training set')
            elif left_size_train > 0.8 *  inputs_inside.shape[0]:
                raise ValueError('The size of the training set is larger than the dataset') 
            else:
                X_train_1, X_test, y_train_1, y_test = train_test_split(inputs_inside, outputs_inside, test_size=0.2, random_state=42 + run)
                X_train_1, y_train_1 = sampling_data(X_train_1, y_train_1, left_size_train,run = run, method = size_sampling)
            
        else:
            X_train_1, X_test, y_train_1, y_test = train_test_split(inputs_inside, outputs_inside, test_size=0.2, random_state=42+ run)
    # Check if the size of the training set is the same as the number of edges        
        if size_train == inputs_edges.shape[0]:
            X_train = inputs_edges
            y_train = outputs_edges
        else:
            X_train = pd.concat([X_train_1, inputs_edges], ignore_index=True)
            y_train = pd.concat([y_train_1, outputs_edges], ignore_index=True)

    else:
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42+run)
        if size_train != 'all':
            if size_train > X_train.shape[0]:
                raise ValueError('The size of the training set is larger than the dataset')
            else:
                X_train, y_train = sampling_data(X_train, y_train, size_train, run = run, method = size_sampling)
       # else:
        #    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    # Create the model
    # ----------------
    if type_model == 'gradient_boosting':
        param_grid = {
        'n_estimators': [150,200,300], # Number of trees in the forest
        'max_depth': [5,6,7], # Maximum depth of the tree
        'learning_rate': np.linspace(0.05,0.2,5), # Learning rate shrinks the contribution of each tree
        'subsample': np.arange(0.2,1.1,0.2), 
        #'ccp_alpha': np.arange(0, 1.1, 0.1), # Complexity parameter used for Minimal Cost-Complexity Pruning
    }
        model_base = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(model_base, param_grid,
                    cv = 10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
        model = grid_search.best_estimator_

    elif type_model == 'random_forest':
        param_grid = {
        'n_estimators': [150,200,300], # Number of trees in the forest
        'max_depth': [10,20,30], # Maximum depth of the tree
        'ccp_alpha': np.arange(0, 1.1, 0.1), # Complexity parameter used for Minimal Cost-Complexity Pruning
    }

        model_base = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model_base, param_grid,
                    cv = 8, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
        model = grid_search.best_estimator_

    elif type_model == 'polynomial':
        model_base = make_pipeline(PolynomialFeatures(), Ridge())
        param_grid = {
            'polynomialfeatures__degree': np.arange(1,17),
            'ridge__alpha': np.logspace(-12, 5, 18)
        }
        grid_search = GridSearchCV(model_base, param_grid,
                    cv = 10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
        model = grid_search.best_estimator_
    elif type_model == 'polynomial_no_tuning':
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha = 0.0001))
        model.fit(X_train, y_train)
    # Predict the values
    # -------------------
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2 score for test: {r2}")
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train,y_train_pred)
    print(f"R2 score for train:{r2_train}")
    # Save the model
    # --------------
    if save == True:
        save_model(model, f'regression/models_{type_model}/{output}/model_{output}.pkl')
        save_model((X_test, y_test), f'regression/models_{type_model}/{output}/test_data.pkl')
        save_model((X_train, y_train), f'regression/models_{type_model}/{output}/train_data.pkl')
        save_model((inputs, outputs), f'regression/models_{type_model}/{output}/total_data.pkl')
    return X_train, mse,  r2
