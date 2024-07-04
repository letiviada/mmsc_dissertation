import json
import pandas as pd

def get_data_from_json(filename:str) -> pd.DataFrame:
    """
    Function that gets the json file and processes the data

    Parameters:
    ----------
    filename (str): the path to the json file

    Returns:
    -------
    ml_data (pd.DataFrame): the data in a pandas DataFrame
    """
    with open(filename) as f:
        data_json = json.load(f)

    data_list = []
    for key, value in data_json.items():
        _, alpha_beta = key.split('=')
        keys_str = alpha_beta.strip("()").split(',')
        alpha,beta = float(keys_str[0]), float(keys_str[1])
        record = {'alpha': alpha, 'beta': beta}
        record.update(value)
        data_list.append(record)

    ml_data = pd.DataFrame(data_list)
    return ml_data

def get_input_output(ml_data: pd.DataFrame, ouput_name:str) -> pd.DataFrame:
    """
    Function that gets the columns of the data

    Parameters:
    ----------
    ml_data (pd.DataFrame): the data in a pandas DataFrame

    Returns:
    -------
    input (pd.DataFrame): the input data of the regression
    output (pd.DataFrame): the performance metric we are measuring
    """
    data = ml_data[['alpha', 'beta', ouput_name]]
    return data
