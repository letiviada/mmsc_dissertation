import json
import pandas as pd
import numpy as np
import os
from utils_r.help_functions import create_interp
from scipy.integrate import quad
from scipy.stats.qmc import LatinHypercube, scale
from sklearn.neighbors import NearestNeighbors

def sampling_data(X, y, size:int, run = 0, method:str='random'):
    if method == 'random':
        X_new = X.sample(size, random_state = run)
        y_new = y.loc[X_new.index]
    elif method == 'latin_hypercube':
        size = int(size)
        unique_points = pd.DataFrame()
    
        while len(unique_points) < size:
        # Define the Latin Hypercube sampler
            lhs_scipy = LatinHypercube(d=2, seed = run)
            sample = lhs_scipy.random(n=size * 2)  # Generate more points than needed to ensure we get enough unique points
            l_bounds = np.array([X['adhesivity'].min(), X['particle_size'].min()])
            u_bounds = np.array([X['adhesivity'].max(), X['particle_size'].max()])
            sample_scaled = scale(sample, l_bounds, u_bounds)
            
            # Create a DataFrame from the scaled sample
            sample_df = pd.DataFrame(sample_scaled, columns=['adhesivity', 'particle_size'])
            
            # Find the nearest neighbors in the original dataset
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X[['adhesivity', 'particle_size']])
            distances, indices = nbrs.kneighbors(sample_df)
            closest_points = X.iloc[indices.flatten()]
        
        # Combine and ensure the points are unique
            unique_points = pd.concat([unique_points, closest_points]).drop_duplicates(subset=['adhesivity', 'particle_size'])
        unique_points = unique_points.head(size)
        
        X_new = unique_points
        #print(X_new.columns)
        y_new = y.loc[X_new.index]
        #print(size, y_new.shape)

    return X_new, y_new

def data_time_old(time:int, names:list, data: pd.DataFrame) -> pd.DataFrame:
    """
    Function that creates a new column in the data for the time specified in the names columns

    Parameters:
    ----------
    time (int): the time we want to consider
    names (list): the list of names of the columns we want to consider
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new columns
    """
    #data = get_data_from_json(filename)
    filter_working_indices = data[data['termination_time'] > time].index
    filter_finished_indices = data[data['termination_time'] <= time].index
    for name in names:
        if name == 'volume_liquid':
            data.loc[filter_finished_indices,f'{name}_time_{time}'] = data.loc[filter_finished_indices, 'lifetime']
            for index in filter_working_indices:
                row = data.loc[index]
                interp_func = create_interp(row,'time', 'throughput')
                data.at[index, f'{name}_time_{time}'] = interp_func(time) if interp_func is not None else np.nan
        elif name == 'total_concentration':
            data.loc[filter_finished_indices, f'{name}_time_{time}'] = data.loc[filter_finished_indices, 'efficiency'] 
            # Calculate total concentrratiton processed
            for index in filter_working_indices:
                row = data.loc[index]
                interp_func = create_interp(row, 'time','efficiency_time')
                if row['adhesivity'] == 0.0:
                    data.at[index, f'{name}_time_{time}'] = 0.0
                else:
                    data.at[index, f'{name}_time_{time}'] = interp_func(time) if interp_func is not None else np.nan
    return data
