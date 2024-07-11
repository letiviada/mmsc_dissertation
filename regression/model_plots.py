from sklearn.metrics import mean_squared_error, r2_score
from utils import open_model, open_data_model
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import scatter_solutions

def model_eval(output_variable,type_eval):
    # Load model and data
    model = open_model(output_variable, model_path='regression/models/')
    inputs, outputs = open_data_model(type_eval, output_variable)
    # Predict the values
    y_pred = model.predict(inputs)
    outputs = outputs.to_frame()
    outputs['Prediction'] = y_pred.tolist()
    mse = mean_squared_error(outputs[output_variable], outputs['Prediction']) 
    r2 = r2_score(outputs[output_variable], outputs['Prediction'])
    print(f"Mean squared error, {type_eval} : {mse}")
    print(f"R2 score,{type_eval} : {r2}")
    return inputs, outputs
    
def main(output_variable, type_eval):  
    inputs, outputs = model_eval(output_variable,type_eval)
    print(inputs.head(), outputs.head())
    scatter_solutions(inputs,outputs,output_variable)
    plt.show()

if __name__ == '__main__':
    names =['total']
    output_variable = 'volume_liquid_time_400'
    for name_eval in names:
        main(output_variable, name_eval) 





