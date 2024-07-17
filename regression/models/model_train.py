from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import save_model, obtain_data, sampling_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def ensure_alpha_inclusion(data, beta_col, alpha_col):
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

def train_model(output,data, size_train = 'all',type_model = 'random_forest',save = True):
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
        print(inputs_edges.shape)

    # Check if the training set is the whole dataset
        if size_train != 'all':
            left_size_train =  size_train - inputs_edges.shape[0]
            if size_train < inputs_edges.shape[0]:
                raise ValueError('The size of the points in the edges is larger than the training set')
            elif left_size_train > inputs_inside.shape[0]:
                raise ValueError('The size of the training set is larger than the dataset') 
            else:
                X_train_1, X_test, y_train_1, y_test = train_test_split(inputs_inside, outputs_inside, test_size=0.2, random_state=42)
                X_train_1, y_train_1 = sampling_data(X_train_1, y_train_1, left_size_train)
            
        else:
            X_train_1, X_test, y_train_1, y_test = train_test_split(inputs_inside, outputs_inside, test_size=0.2, random_state=42)
    # Check if the size of the training set is the same as the number of edges        
        if size_train == inputs_edges.shape[0]:
            X_train = inputs_edges
            y_train = outputs_edges
        else:
            X_train = pd.concat([X_train_1, inputs_edges], ignore_index=True)
            y_train = pd.concat([y_train_1, outputs_edges], ignore_index=True)

    else:
        if size_train != 'all':
            if size_train > X_train.shape[0]:
                raise ValueError('The size of the training set is larger than the dataset')
            else:
                X_train, y_train = sampling_data(X_train, y_train, size_train)
        else:
            X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    # Create the model
    # ----------------
    if type_model == 'gradient_boosting':
        param_grid = {
        'n_estimators': [150,200,300], # Number of trees in the forest
        'max_depth': [5,10,20], # Maximum depth of the tree
        'learning_rate': [0.01, 0.1, 1], # Learning rate shrinks the contribution of each tree
        'ccp_alpha': np.arange(0, 1.1, 0.1), # Complexity parameter used for Minimal Cost-Complexity Pruning
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
                    cv = 10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
        model = grid_search.best_estimator_

    elif type_model == 'polynomial':
        model_base = make_pipeline(PolynomialFeatures(), Ridge())
        param_grid = {
            'polynomialfeatures__degree': np.arange(10,15),
            'ridge__alpha': np.logspace(-10, 0, 11)
        }
        grid_search = GridSearchCV(model_base, param_grid,
                    cv = 10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
        model = grid_search.best_estimator_
    # Predict the values
    # -------------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2 score for test: {r2}")

    # Save the model
    # --------------
    if save == True:
        save_model(model, f'regression/models_{type_model}/{output}/model_{output}.pkl')
        save_model((X_test, y_test), f'regression/models_{type_model}/{output}/test_data.pkl')
        save_model((X_train, y_train), f'regression/models_{type_model}/{output}/train_data.pkl')
        save_model((inputs, outputs), f'regression/models_{type_model}/{output}/total_data.pkl')
    return mse,  r2