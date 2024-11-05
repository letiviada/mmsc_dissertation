import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Add path for additional modules
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
from utils_r import open_model, open_data_model

def model_eval(output_variable: str, type_eval: str, type_model: str):
    """
    Evaluates the ML model using mean squared error and R2 score.

    Parameters
    ----------
    output_variable : str
        The output variable to be predicted.
    type_eval : str
        The type of evaluation to perform (total, train, or test).
    type_model : str
        The type of model to be used (random_forest, gradient_boosting, polynomial).

    Returns
    -------
    inputs : pd.DataFrame
        The input data (adhesivity and particle size).
    outputs : pd.DataFrame
        The output data (output variable and predictions).
    """
    # Load model and data
    model_path = f'regression/models_{type_model}/'
    model = open_model(output_variable, model_path=model_path)
    inputs, outputs = open_data_model(type_eval, output_variable, model_path=model_path)
    
    # Make predictions
    y_pred = model.predict(inputs)
    outputs = outputs.to_frame()
    outputs['prediction'] = y_pred.tolist()
    
    # Calculate metrics
    mse = mean_squared_error(outputs[output_variable], outputs['prediction']) 
    r2 = r2_score(outputs[output_variable], outputs['prediction'])
    
    print(f"Mean Squared Error ({type_eval}): {mse}")
    print(f"R2 Score ({type_eval}): {r2}")
    
    return inputs, outputs