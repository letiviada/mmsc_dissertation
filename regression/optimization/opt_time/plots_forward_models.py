import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/multiscale')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_time_opt, save_figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the data
data_physical = pd.read_csv('regression/optimization/opt_time/data/physical_large/data_varying_n_min.csv')
data_ml = pd.read_csv('regression/optimization/opt_time/data/ml_range/data_varying_n_min.csv')


particle_size = 0.04
weight_coefficient = [1.0,1.2,1.4]

#data_physical.loc[:,'gamma'] = 400 * data_physical.loc[:,'volume_time_400_scaled']
#data_ml.loc[:,'gamma'] =  400*data_ml.loc[:,'volume_time_400_scaled']

fig, ax = plot_time_opt(data_physical, particle_size, weight_coefficient, data_ml=data_ml, save = False)
save_figure(fig, 'regression/optimization/opt_time/plots/underestimation')
plt.show()

data_opt = pd.read_csv('regression/optimization/opt_time/data/ml/optimum_values.csv')
data_physical = pd.read_csv('regression/optimization/opt_time/data/physical_large/data_varying_n_min.csv')
data_opt_ps = data_opt[data_opt['particle_size'] == particle_size]
data_physical_ps = data_physical[data_physical['particle_size'] == particle_size]
#data_physical_ps = data_physical_ps[data_physical_ps['weight_coefficient'].isin(weight_coefficient)]
# Show all rows of the dataframe
pd.set_option('display.max_rows', None)
#print(data_opt_ps)


# Plot the product as a function of alpha
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].plot(filtered_data['adhesivity'], filtered_data['product'])
#ax[1].plot(filtered_data2['adhesivity'], filtered_data2['product'])
#plt.xlabel('Alpha')
#plt.ylabel('Product')
#plt.title(f'Product vs Alpha for Particle Size {particle_size_value}')
#plt.show()