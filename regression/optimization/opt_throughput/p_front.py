import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import pareto_front, save_figure
import numpy as np
import matplotlib.pyplot as plt
particle_size = np.arange(0.02, 0.11, 0.02)
# Load the data
data_initial = pd.read_csv('regression/optimization/opt_throughput/data/ml_range/data_for_sums.csv')
data_marker = pd.read_csv('regression/optimization/opt_throughput/data/physical/data_for_sums.csv')
# Change data
data_initial.loc[data_initial['adhesivity'] == 0, 'time_throughput_100_scaled'] = 1
data_initial.loc[data_initial['adhesivity'] == 0, 'efficiency_throughput_100'] = 0
#save_data_to_csv(data_initial, 'optimization/opt_time/data/ml','data_for_sums.csv')
data_initial.loc[:, 'time_throughput_100'] = 100 / data_initial['time_throughput_100_scaled'] 
data_marker.loc[:, 'time_throughput_100'] = 100 / data_marker['time_throughput_100_scaled']
data_initial = data_initial[data_initial['particle_size'].isin(particle_size)]
fig, ax = pareto_front(data_initial,  'time_throughput_100','efficiency_throughput_100',
                        min_throughput_value = 200, min_efficiency_value = 0.4, adhesivity='all', save = False)
ax[0].set_ylim(0, 1.0)
ax[0].set_xlim(99, 300)
ax[0].set_xlabel(r'$T(\theta=100)$')
ax[0].set_ylabel(r'$\eta(\theta=100)$')
save_figure(fig, 'regression/optimization/opt_throughput/plots/pareto_front')
plt.show()