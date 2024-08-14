import pandas as pd
import sys

import numpy as np
data_before = pd.read_csv('regression/optimization/opt_time/data/ml_range/optimum_values.csv')

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
        """
        Function that preprocesses the data.

        Parameters:
        ----------
        data (pd.DataFrame): the data we want to preprocess

        Returns:
        -------
        data (pd.DataFrame): the preprocessed data
        """
        # Group by particle size and weight coefficient, calculate mean and std of adhesivity_time_400
        grouped_data = data.groupby(['particle_size', 'weight_coefficient'])['adhesivity_time_400'].agg(['mean', 'std']).reset_index()

        # Create a new dataframe with columns particle size, weight coefficient, mean, and std
        new_data = pd.DataFrame({
            'particle_size': grouped_data['particle_size'],
            'weight_coefficient': grouped_data['weight_coefficient'],
            'mean': grouped_data['mean'],
            'std': grouped_data['std']
        })

        return new_data
pd.set_option('display.max_rows', None)
data_after = preprocess(data_before)
print(data_after)


   