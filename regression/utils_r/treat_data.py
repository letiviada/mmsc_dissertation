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
        y_new = y.loc[X_new.index]

    return X_new, y_new
