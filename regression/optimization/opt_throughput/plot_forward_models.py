import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/multiscale')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_time_opt, plot_one_weight
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the data
data_physical = pd.read_csv('regression/optimization/opt_throughput/data/physical/data_varying_n_min.csv')
data_ml = pd.read_csv('regression/optimization/opt_throughput/data/ml/data_varying_n_min.csv')

data_physical.rename(columns={'adhesivity_throughput_100': 'adhesivity'}, inplace=True)
data_ml.rename(columns={'adhesivity_throughput_100': 'adhesivity'}, inplace=True)

particle_size = [0.02, 0.04, 0.06, 0.08, 0.1]
weight_coefficient = 1.0

fig, ax = plot_one_weight(data_physical, weight_coefficient,particle_size, data_ml=data_ml, save = True)
plt.show()