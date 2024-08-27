# Plot the range of values before doing ML on (beta, n) -> range
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_range_throughput, save_figure
from scipy.ndimage import gaussian_filter1d
increased_sigma = 1.5
data_pd = pd.read_csv('regression/optimization/opt_throughput/data/physical/optimum_values.csv')
data_ml = pd.read_csv('regression/optimization/opt_throughput/data/ml_range/optimum_values.csv')
data_ml['adhesivity_throughput_100'] = data_ml.groupby('particle_size')['adhesivity_throughput_100'].transform(
    lambda x: gaussian_filter1d(x, sigma=increased_sigma)
)
data_pd['adhesivity_throughput_100'] = data_pd.groupby('particle_size')['adhesivity_throughput_100'].transform(
    lambda x: gaussian_filter1d(x, sigma=increased_sigma)
)
throughput = 100
particle_size = np.arange(0.04, 0.11, 0.02).round(2)
fig, ax = plot_range_throughput(data_pd, throughput, particle_size, data_ml,save=False) 
ax[0].set_yticks(np.arange(0,1.1,0.2))
save_figure(fig, 'regression/optimization/opt_throughput/plots/range_throughput')
plt.show()