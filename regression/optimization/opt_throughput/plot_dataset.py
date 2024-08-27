import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression/')
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
from utils_r import open_model
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_for_spec_throughput, plot_for_varying_beta
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_initial = pd.read_csv('regression/optimization/opt_throughput/data/ml/data_for_sums.csv')
data_initial_ps = pd.read_csv('regression/optimization/opt_throughput/data/ml_ps/data_for_sums.csv')
data_marker = pd.read_csv('regression/optimization/opt_throughput/data/throughput_100/initial_dataset.csv')
fig, ax = plot_for_spec_throughput(data_initial, 'time_throughput_100', 
                                   particle_size=[0.02, 0.04,0.06, 0.08, 0.1], data_marker= data_marker, save = False)
#fig, ax = plot_for_varying_beta(data_initial_ps, 'efficiency_throughput_100',[0.2,0.4,0.6,0.8,1.0],data_marker,save = False)

plt.show()


