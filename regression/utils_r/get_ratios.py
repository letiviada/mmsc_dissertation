import pandas as pd
import numpy as np

def get_ratio(numerator: str, denominator: str, power: float, data: pd.DataFrame) -> pd.DataFrame:
    
    """
    Function that gets the ratio of two columns in the data

    Parameters:
    ----------
    numerator (str): the name of the column we want to consider as the numerator
    denominator (str): the name of the column we want to consider as the denominator
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new column
    """
    ratio =  pd.DataFrame(
        {'adhesivity': data.loc[:, 'adhesivity'],
            'particle_size': data.loc[:, 'particle_size'],
            numerator: data.loc[:, numerator],
            denominator: data.loc[:, denominator],
        'ratio':(data.loc[:, numerator] ** power) / (data.loc[:, denominator])}
    )
    return ratio
def get_product(numerator: str, denominator: str, power: float, data: pd.DataFrame) -> pd.DataFrame:
    
    """
    Function that gets the ratio of two columns in the data

    Parameters:
    ----------
    numerator (str): the name of the column we want to consider as the numerator
    denominator (str): the name of the column we want to consider as the denominator
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new column
    """
    ratio =  pd.DataFrame(
        {'adhesivity': data.loc[:, 'adhesivity'],
            'particle_size': data.loc[:, 'particle_size'],
            numerator: data.loc[:, numerator],

            denominator: data.loc[:, denominator],
        'product':(data.loc[:, numerator] ** power) * (data.loc[:, denominator])}
    )
    return ratio
def weight_sum(data:pd.DataFrame, function1:str, function2:str, weight:float) -> pd.DataFrame:
    """
    Function that calculates the weighted sum of two functions

    Parameters:
    -----------
    data (pd.DataFrame): the data we want to consider
    function1 (str): the name of the first function
    function2 (str): the name of the second function
    weight (float): the weight we want to consider

    Returns:
    --------
    data (pd.DataFrame): the data with the new column
    """
    weight_sum = pd.DataFrame(
        {'adhesivity': data.loc[:, 'adhesivity'],
         'particle_size': data.loc[:, 'particle_size'],
            function1: data.loc[:, function1],
            function2: data.loc[:, function2],
        'product': weight * data.loc[:, function1]  + (1-weight) *(data.loc[:, function2])}
    )
    return weight_sum

def get_product_time(numerator1: str, numerator2: str, denominator: str, power: float, data: pd.DataFrame) -> pd.DataFrame:
    
    """
    Function that gets the ratio of two columns in the data

    Parameters:
    ----------
    numerator (str): the name of the column we want to consider as the numerator
    denominator (str): the name of the column we want to consider as the denominator
    data (pd.DataFrame): the data we want to consider

    Returns:
    -------
    data (pd.DataFrame): the data with the new column
    """
    ratio =  pd.DataFrame(
        {'adhesivity': data.loc[:, 'adhesivity'],
            'particle_size': data.loc[:, 'particle_size'],
            numerator1: data.loc[:, numerator1],
            numerator2: data.loc[:, numerator2],
            denominator: data.loc[:, denominator],
        'product':(data.loc[:, numerator1] ** power) * (data.loc[:,numerator2]) / (data.loc[:, denominator])}
    )
    return ratio

def ratio_predictions(numerator_predictions, denominator_predictions, n):
    """
    Function that calculates the ratio predictions

    Parameters:
    -----------
    volume_predictions (np.array): the volume predictions
    concentration_predictions (np.array): the concentration predictions
    n (int): the power to which the volume predictions are raised

    Returns:
    --------
    ratio (np.array): the ratio predictions
    """
    return (numerator_predictions ** n)/(denominator_predictions)
