import os
import joblib
from utils.treat_data import obtain_data, clean_data
def save_model(model, model_path):
    """
    Save the ML model

    Parameters:
    ----------
    model (object): the model to be saved
    model_path (str): the path to save the model
    
    """
    model_dir = os.path.dirname(model_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

def open_model(output, model_path):
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

def open_data_model(type_eval, output, model_path):
    """
    Open data for the model evaluation

    Parameters:
    ----------
    type_eval (str): the type of data to open, either 'total', 'train' or 'test'
    output (str): the output variable to predict
    model_path (str): the path to the model (only fill in if type_eval != 'total')

    Returns:
    --------
    inputs (pd.DataFrame): the input data

    """

    output_data = output.replace(' ', '_').lower()
    path_to_data  = os.path.join(model_path, f'{output}/{type_eval}_data.pkl')
    inputs, outputs = joblib.load(path_to_data)
    return inputs, outputs

def save_data_to_csv(data, filename):
    """
    Save data to a csv file

    Parameters:
    ----------
    data (pd.DataFrame): the data to be saved
    filename (str): the name of the file to save the data

    """
    directory = '/home/viadacampos/Documents/mmsc_dissertation/regression/sample_size_study/'
    filepath = os.path.join(directory, filename)
    if not os.path.exists(directory):
        directory = '/Users/letiviada/dissertation_mmsc/regression/sample_size_study/'
        filepath = os.path.join(directory, filename)
        if not os.path.exists(directory):
            raise FileNotFoundError("The directories do not exist")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    data.to_csv(filepath, index = False)
    print(f"Data saved")