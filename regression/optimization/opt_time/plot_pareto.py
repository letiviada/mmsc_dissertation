import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression/')
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
from utils_r import open_model, save_data_to_csv
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_for_spec_throughput, plot_for_varying_beta, make_loglog, pareto_front
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
particle_size = np.arange(0.02, 0.11, 0.02)
# Load the data
data_initial = pd.read_csv('regression/optimization/opt_time/data/ml_range/data_for_sums.csv')
data_marker = pd.read_csv('regression/optimization/opt_time/data/physical/data_for_sums.csv')
data_marker_nc = pd.read_csv('regression/optimization/opt_time/data/physical/time_0/data_for_sums.csv')
# Change data
data_initial.loc[data_initial['adhesivity'] == 0, 'volume_time_400_scaled'] = 1
data_initial.loc[data_initial['adhesivity'] == 0, 'efficiency_time_400'] = 0
#save_data_to_csv(data_initial, 'optimization/opt_time/data/ml','data_for_sums.csv')
data_initial.loc[:, 'throughput_time_400'] = data_initial['volume_time_400_scaled'] * 400
data_marker.loc[:, 'throughput_time_400'] = data_marker['volume_time_400_scaled'] * 400
data_initial = data_initial[data_initial['particle_size'].isin(particle_size)]
fig, ax = pareto_front(data_initial,  'throughput_time_400','efficiency_time_400',
                        min_throughput_value = 100, min_efficiency_value = 0.5, adhesivity='all', save = False)
ax[0].set_ylim(0, 1.0)
ax[0].set_xlim(0, 400)
plt.show()