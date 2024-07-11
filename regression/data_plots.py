from utils import obtain_data, get_data_from_json, clean_data, data_time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import view_data_all_outputs, boxplot

def dataplots(outputs):
    """
    Function that plots figures for the data we have obtained

    Parameters:
    ----------
    outputs (list): List of output parameters we are interested in
    """
    data = get_data_from_json(
    filename = 'performance_indicators/performance_indicators_phi_1.0.json')
    print(data.describe())
    view_data_all_outputs(data,outputs)
    plt.show()
    

if __name__ == '__main__':
    outputs =['Termination time', 'Lifetime'] 
    dataplots(outputs)
