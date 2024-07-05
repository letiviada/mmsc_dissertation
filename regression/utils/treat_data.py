import json
import pandas as pd

def get_data_from_json(filename:str = '/home/viadacampos/mmsc_dissertaion/multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json') -> pd.DataFrame:
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
        record = {'Adhesivity': alpha, 'Particle Size': beta}
        record.update(value)
        data_list.append(record)

    ml_data = pd.DataFrame(data_list)
    data_to_keep = ml_data[['Adhesivity', 'Particle Size', 'Termination time', 'Lifetime']] 
    return ml_data, data_to_keep


def obtain_data(data_all, output:str, filename: str = '/home/viadacampos/Documents/mmsc_dissertation/multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json') -> pd.DataFrame:
  
    data = data_all[['Adhesivity', 'Particle Size', output]]
    return data

def clean_data(filename: str = '/home/viadacampos/Documents/mmsc_dissertation/multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json'):
    _, data_to_keep = get_data_from_json(filename)
    data_to_keep = data_to_keep[data_to_keep['Lifetime'] <= 200]
        # Just to check if the value i wanted to remove did it
    #data.sort_values(by=['alpha', 'beta'], inplace=True)
    #print(data)
    return data_to_keep


