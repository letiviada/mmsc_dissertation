import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression/')
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
from utils_r import open_model
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_for_spec_throughput
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_initial = pd.read_csv('regression/optimization/opt_throughput/data/ml/data_for_sums.csv')
data_marker = pd.read_csv('regression/optimization/opt_throughput/data/throughput_100/initial_dataset.csv')
fig, ax = plot_for_spec_throughput(data_initial, 'time_throughput_100', particle_size=[0.1], data_marker= data_marker, save = False)
plt.show()

