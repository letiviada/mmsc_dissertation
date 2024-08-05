import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_r import get_data_from_json
import sys 
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_perf_ind

data = get_data_from_json('performance_indicators/performance_indicators_sample_size.json')
data = data[data['adhesivity'] == 0.6]

#data_plot = data[['adhesivity', 'particle_size', 'efficiency']]
data_plot = data.sort_values('particle_size')
# def plot_perf_ind(data:pd.DataFrame, output:str, input:str, save:bool):
#fig, ax = plot_perf_ind(data_plot, 'efficiency', 'adhesivity', save=True)
fig, ax = plot_perf_ind(data_plot, 'termination_time', 'particle_size', save=False)
plt.show()