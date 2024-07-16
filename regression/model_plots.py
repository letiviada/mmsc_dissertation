import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from regression.models.model_eval import model_eval
from multiscale.plotting import scatter_solutions

    
def main(output_variable, type_eval, type_model):  
    inputs, outputs = model_eval(output_variable,type_eval,type_model)
    fig, ax = scatter_solutions(inputs,outputs,output_variable, type_model)
    plt.show()

if __name__ == '__main__':
    names =['train','test','total']
    #output_variable = 'volume_liquid_time_400'
    output_variable = 'lifetime'
    for name_eval in names:
        main(output_variable, name_eval,type_model = 'polynomial') 





