import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from regression.models.model_eval import model_eval
from multiscale.plotting import scatter_solutions
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
from utils import open_model
import numpy as np
import pandas as pd
from itertools import product

    
def main(output_variable, type_eval, type_model):  
    inputs1, outputs1 = model_eval(output_variable,type_eval,type_model)
    adhesivity_range = np.arange(0.3, 1.1, 0.01).round(2)
    particle_size_range = np.arange(0.03, 0.1, 0.001).round(2)
    # Generate all combinations of 'adhesivity' and 'particle_size'
    all_combinations = list(product(adhesivity_range, particle_size_range))
    # Create DataFrame from combinations
    inputs = pd.DataFrame(all_combinations, columns=['adhesivity', 'particle_size'])
    model = open_model(output_variable, model_path=f'regression/models_{type_model}/')
    y_pred = model.predict(inputs)
    y_pred_df = pd.DataFrame(y_pred, columns=[output_variable])
    final_df = pd.concat([inputs, y_pred_df], axis=1)
    true_df =pd.concat([inputs1, outputs1], axis=1)
    fig, ax = scatter_solutions(inputs1,outputs1,output_variable, type_model)
    ax[1].plot(final_df[final_df['particle_size'] == 0.08]['adhesivity'], final_df[final_df['particle_size'] == 0.08][output_variable])
    plt.show()
if __name__ == '__main__':
    names =['total']
    output_variable = 'lifetime'
    for name_eval in names:
        main(output_variable, name_eval,type_model = 'polynomial') 