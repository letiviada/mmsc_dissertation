# Plot the range of values before doing ML on (beta, n) -> range
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_range_time
increased_sigma = 2.5

data_pd = pd.read_csv('regression/optimization/opt_time/data/physical_large/optimum_values.csv')
data_ml = pd.read_csv('regression/optimization/opt_time/data/ml_range/optimum_values.csv')
data_ml['adhesivity_time_400'] = data_ml.groupby('particle_size')['adhesivity_time_400'].transform(
    lambda x: gaussian_filter1d(x, sigma=increased_sigma)
)
data_pd['adhesivity_time_400'] = data_pd.groupby('particle_size')['adhesivity_time_400'].transform(
    lambda x: gaussian_filter1d(x, sigma=increased_sigma)
)
time = 400
particle_size = np.arange(0.04, 0.11, 0.02).round(2)
particle_size = [0.04, 0.06, 0.08, 0.1]
fig, ax = plot_range_time(data_pd, time, particle_size, data_ml = data_ml, save=True)  
plt.show()