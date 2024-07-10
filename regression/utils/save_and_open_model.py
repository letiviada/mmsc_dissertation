import os
import joblib
from utils.treat_data import obtain_data, clean_data
def save_model(model, model_path):
    model_dir = os.path.dirname(model_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

def open_model(output, model_path = 'regression/models/'):
    """
    Open the ML model

    Parameters:
    ----------
    model (str): the name of the model
    model_path (str): the path to the model

    """
    path_to_model = os.path.join(model_path, f'{output}/model_{output}.pkl')
    model = joblib.load(path_to_model)
    return model

def open_data_model(type_eval, output, filename = 'performance_indicators_phi_1.0.json', model_path = 'regression/models/'):
    """
    Open data for the model evaluation

    Parameters:
    ----------
    type_eval (str): the type of data to open, either 'total', 'train' or 'test'
    output (str): the output variable to predict
    filename (str): the name of the file to open (only fill in if type_eval == 'total')
    model_path (str): the path to the model (only fill in if type_eval != 'total')

    Returns:
    --------
    inputs (pd.DataFrame): the input data

    """
    if type_eval == 'total':
        data = obtain_data(output, filename)
        inputs, outputs = data.drop(output, axis = 1), data[output]
    else:
        path_to_data  = os.path.join(model_path, f'{output}/{type_eval}_data.pkl')
        inputs, outputs = joblib.load(path_to_data)
    return inputs, outputs