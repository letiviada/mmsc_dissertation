import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add custom paths for additional modules
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')

from utils_r import save_model, obtain_data, sampling_data

def ensure_alpha_inclusion(data: pd.DataFrame, beta_col: str, alpha_col: str):
    """
    Ensures the training dataset includes rows with min and max alpha values for each unique beta.
    
    Parameters:
    -----------
    data (pd.DataFrame): Dataset to be evaluated
    beta_col (str): Column containing beta values
    alpha_col (str): Column containing alpha values
    
    Returns:
    --------
    edge_rows (pd.DataFrame): Rows with min and max alpha values for each beta
    remaining_data (pd.DataFrame): Data without the mandatory edge rows
    """
    unique_betas = data[beta_col].unique()
    edge_rows = pd.DataFrame()
    indices_to_drop = []
    
    for beta in unique_betas:
        beta_data = data[data[beta_col] == beta]
        min_alpha_index = beta_data[alpha_col].idxmin()
        max_alpha_index = beta_data[alpha_col].idxmax()
        edge_rows = pd.concat([edge_rows, beta_data.loc[[min_alpha_index, max_alpha_index]]], ignore_index=True)
        indices_to_drop.extend([min_alpha_index, max_alpha_index])

    remaining_data = data.drop(indices_to_drop)
    return edge_rows, remaining_data

def train_model(output: str, data: pd.DataFrame, size_train='all', run=0, size_sampling='random', type_model='random_forest', save=True):
    """
    Trains a model for a specified output and dataset, tunes hyperparameters, and optionally saves the model.
    
    Parameters:
    -----------
    output (str): Output variable name
    data (pd.DataFrame): Dataset
    size_train (int or str): Size of the training set, default is 'all'
    run (int): Random seed for reproducibility
    size_sampling (str): Sampling method, default is 'random'
    type_model (str): Model type ('random_forest', 'gradient_boosting', 'polynomial', 'polynomial_no_tuning')
    save (bool): Whether to save the model, default is True
    
    Returns:
    --------
    X_train (pd.DataFrame): Training input features
    mse (float): Mean squared error of the test set
    r2 (float): R-squared of the test set
    """
    inputs, outputs = data.drop(columns=[output]), data[output]
    
    if type_model == 'polynomial':
        edges, inside = ensure_alpha_inclusion(data, 'particle_size', 'adhesivity')
        inputs_inside, outputs_inside = inside.drop(columns=[output]), inside[output]
        inputs_edges, outputs_edges = edges.drop(columns=[output]), edges[output]

        # Split and sample based on training set size
        if size_train != 'all':
            left_size_train = size_train - inputs_edges.shape[0]
            if size_train < inputs_edges.shape[0]:
                raise ValueError('Training set size is smaller than the edge data size.')
            elif left_size_train > 0.8 * inputs_inside.shape[0]:
                raise ValueError('Training set size exceeds available dataset size.')
            else:
                X_train_1, X_test, y_train_1, y_test = train_test_split(inputs_inside, outputs_inside, test_size=0.2, random_state=42 + run)
                X_train_1, y_train_1 = sampling_data(X_train_1, y_train_1, left_size_train, run=run, method=size_sampling)
        else:
            X_train_1, X_test, y_train_1, y_test = train_test_split(inputs_inside, outputs_inside, test_size=0.2, random_state=42 + run)
        
        # Combine edge rows if needed
        if size_train == inputs_edges.shape[0]:
            X_train, y_train = inputs_edges, outputs_edges
        else:
            X_train = pd.concat([X_train_1, inputs_edges], ignore_index=True)
            y_train = pd.concat([y_train_1, outputs_edges], ignore_index=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42 + run)
        if size_train != 'all' and size_train <= X_train.shape[0]:
            X_train, y_train = sampling_data(X_train, y_train, size_train, run=run, method=size_sampling)

    # Model Selection and Hyperparameter Tuning
    # -----------------------------------------
    if type_model == 'gradient_boosting':
        param_grid = {
            'n_estimators': [600],
            'max_depth': [5, 6, 7],
            'learning_rate': np.linspace(0.05, 0.2, 5),
            'subsample': np.arange(0.2, 1.1, 0.2)
        }
        model_base = GradientBoostingRegressor(random_state=42)
    elif type_model == 'random_forest':
        param_grid = {
            'n_estimators': [150, 200, 300],
            'max_depth': [10, 20, 30],
            'ccp_alpha': np.arange(0, 1.1, 0.1)
        }
        model_base = RandomForestRegressor(random_state=42)
    elif type_model == 'polynomial':
        param_grid = {
            'polynomialfeatures__degree': np.arange(1, 17),
            'ridge__alpha': np.logspace(-12, 5, 18)
        }
        model_base = make_pipeline(PolynomialFeatures(), Ridge())
    elif type_model == 'polynomial_no_tuning':
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.0001))
        model.fit(X_train, y_train)
    else:
        grid_search = GridSearchCV(model_base, param_grid, cv=10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_} with score {grid_search.best_score_}")

    # Evaluation
    # ----------
    y_pred = model.predict(X_test)
    mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
    print(f"Test MSE: {mse}, Test R2: {r2}")
    
    y_pred_train = model.predict(X_train)
    mse_train, r2_train = mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)
    print(f"Train MSE: {mse_train}, Train R2: {r2_train}")

    # Save Model and Data
    # --------------------
    if save:
        save_model(model, f'regression/models_{type_model}/{output}/model_{output}.pkl')
        save_model((X_test, y_test), f'regression/models_{type_model}/{output}/test_data.pkl')
        save_model((X_train, y_train), f'regression/models_{type_model}/{output}/train_data.pkl')
        save_model((inputs, outputs), f'regression/models_{type_model}/{output}/total_data.pkl')
    
    return X_train, mse, r2