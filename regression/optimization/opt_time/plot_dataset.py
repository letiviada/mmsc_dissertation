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

# Load the data
data_initial = pd.read_csv('regression/optimization/opt_time/data/ml/data_for_sums.csv')
data_marker = pd.read_csv('regression/optimization/opt_time/data/physical/data_for_sums.csv')
data_marker_nc = pd.read_csv('regression/optimization/opt_time/data/physical/time_0/data_for_sums.csv')
data_initial.loc[:, 'throughput_time_400'] = data_initial['volume_time_400_scaled'] * 400
data_marker.loc[:, 'throughput_time_400'] = data_marker['volume_time_400_scaled'] * 400
data_marker_nc.loc[:, 'throughput_time_0'] = data_marker_nc['volume_time_0_scaled']
fig, ax = plot_for_spec_throughput(data_initial, 'efficiency_time_400', particle_size=[0.2, 0.4,0.6 ,0.8 ,1.0], data_marker= data_marker, save = True)
fig, ax = plot_for_varying_beta(data_initial, 'efficiency_time_400', adhesivity=[0.2, 0.4,0.6 ,0.8 ,1.0], data_marker= data_marker, save = True)
fig, ax = make_loglog(data_marker, 'throughput_time_400', betas = [0.02,0.04,0.06, 0.08, 0.1], type_data = 'physical',data_compare=data_marker_nc, save = True)
plt.show()
