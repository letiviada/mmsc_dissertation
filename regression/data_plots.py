from utils_r import obtain_data, get_data_from_json, clean_data, data_time
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
    data = clean_data(filename = 'performance_indicators/performance_indicators_standard.json')
    #data = data[(data['particle_size'] == 0.05) | (data['particle_size'] == 0.06)]
    view_data_all_outputs(data,outputs)
    plt.show()

if __name__ == '__main__':
    outputs = ['termination_time']# ['termination_time', 'total_throughput','efficiency'] 
    dataplots(outputs)
