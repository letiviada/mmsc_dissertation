import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_r import get_data_from_json
import sys 
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_perf_ind, plot_perf_ind_various_bet, make_loglog, plot_perf_ind_time#

data = get_data_from_json('performance_indicators/performance_indicators_phi_4.0.json')
#data = data[data['adhesivity'] == 0.6]
#data = data[data['particle_size'] == 0.06]
#data_plot = data[['adhesivity', 'particle_size', 'efficiency']]
#data_plot = data.sort_values('particle_size')
#print(data_plot[['particle_size', 'termination_time']])
# def plot_perf_ind(data:pd.DataFrame, output:str, input:str, save:bool):
#fig, ax = plot_perf_ind(data_plot, 'efficiency', 'adhesivity', save=True)
#fig, ax = plot_perf_ind(data_plot, 'termination_time', 'particle_size', save=True)
#data = data.sort_values('adhesivity')
#fig, ax = plot_perf_ind_various_bet(data, 'lifetime', particle_size='all', save=True)
#fig, ax = make_loglog(data, 'termination_time', betas = 'all', type_data = 'physical',data_compare=None, save = False)
#ax[0].set_xticks(np.arange(0.2,1.1,0.2))
#plt.show()
data_throughput = data[['adhesivity', 'particle_size','termination_time', 'lifetime', 'time','throughput']]
data_plot = data_throughput.sort_values('termination_time')
#fix, ax = plot_perf_ind_time(data_plot, output = 'throughput',particle_size=np.arange(0.2,1.2, 0.2).round(2), input_value = 'time', save=True)
plt.show()