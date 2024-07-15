from sklearn.metrics import mean_squared_error, r2_score
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
from utils import open_model, open_data_model

def model_eval(output_variable,type_eval,type_model):
    """
    Function that evaluates the ML model using the mean squared error and the R2 score.

    Parameters:
    ----------
    output_variable (str): the output variable to be predicted
    type_eval (str): the type of evaluation to be done (total, train or test)
    type_model (str): the type of model to be used (random_forest, gradient_boosting, polynomial)

    Returns:
    --------
    inputs (pd.DataFrame): the input data (adhesivity and particle size)
    outputs (pd.DataFrame): the output data (the output variable and the predictions)
    """
    # Load model and data
    model = open_model(output_variable, model_path=f'regression/models_{type_model}/')
    inputs, outputs = open_data_model(type_eval, output_variable, model_path=f'regression/models_{type_model}/')
    # Predict the values
    y_pred = model.predict(inputs)
    outputs = outputs.to_frame()
    outputs['Prediction'] = y_pred.tolist()
    mse = mean_squared_error(outputs[output_variable], outputs['Prediction']) 
    r2 = r2_score(outputs[output_variable], outputs['Prediction'])
    print(f"Mean squared error, {type_eval} : {mse}")
    print(f"R2 score,{type_eval} : {r2}")
    return inputs, outputs