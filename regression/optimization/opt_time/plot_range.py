# Plot the range of values before doing ML on (beta, n) -> range
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_range_time

data_pd = pd.read_csv('regression/optimization/opt_time/data/ml_range/optimum_values.csv')
time = 400
particle_size = np.arange(0.02, 0.1, 0.02)
fig, ax = plot_range_time(data_pd, time, particle_size, save=True) 
plt.show()