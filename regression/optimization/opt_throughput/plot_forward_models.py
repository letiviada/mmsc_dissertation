import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/multiscale')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_time_opt, plot_one_weight, plot_for_spec_throughput, save_figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np



data_physical = pd.read_csv('regression/optimization/opt_throughput/data/physical/data_varying_n_min.csv')
data_ml = pd.read_csv('regression/optimization/opt_throughput/data/ml/data_varying_n_min.csv')
particle_size = 0.06
weight_coefficient = [0.0,0.25,0.5, 1.0]
fig, ax = plot_time_opt(data_physical, particle_size, weight_coefficient, data_ml=data_ml, save = False)
ax[0].set_ylabel(r'$\zeta$')
save_figure(fig, 'regression/optimization/opt_throughput/plots/zeta_small')
plt.show()