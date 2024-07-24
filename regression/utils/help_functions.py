from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

def create_interp(data,x_axis, y_axis):
    x = data[x_axis]
    y = data[y_axis]
    return interp1d(x, y, kind='cubic', fill_value="extrapolate")

def change_name_time(output, time):
    """
    Function to change the name to obtain the one used in the folder

    Parameters:
    ----------
    output (str): the name of the output variable
    time (int): the time we want to consider

    Returns:
    -------
    output_name_folder (str): the name of the folder
    """
    output_name_folder= f'{output}_time_{time}'
    return output_name_folder


def make_data_frame(dataframe, **kwargs):
    for array_name, array in kwargs.items():
        dataframe[array_name] = array
    return dataframe