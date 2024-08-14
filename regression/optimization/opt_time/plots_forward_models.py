import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/multiscale')
sys.path.append('/Users/letiviada/dissertation_mmsc/')
from multiscale.plotting import plot_time_opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the data
data_physical = pd.read_csv('regression/optimization/opt_time/data/physical/data_varying_n_min.csv')
data_ml = pd.read_csv('regression/optimization/opt_time/data/ml/data_varying_n_min.csv')


particle_size = 0.06
weight_coefficient = [8,4,2,1]

fig, ax = plot_time_opt(data_physical, particle_size, weight_coefficient, data_ml=data_ml, save = False)
plt.show()

data_opt = pd.read_csv('regression/optimization/opt_time/data/ml/optimum_values.csv')
data_physical = pd.read_csv('regression/optimization/opt_time/data/ml/data_varying_n_min.csv')
data_opt_ps = data_opt[data_opt['particle_size'] == particle_size]
data_physical_ps = data_physical[data_physical['particle_size'] == particle_size]
data_physical_ps = data_physical_ps[data_physical_ps['weight_coefficient'].isin(weight_coefficient)]
# Show all rows of the dataframe
pd.set_option('display.max_rows', None)
#print(data_opt_ps)
print(data_physical_ps)


# Plot the product as a function of alpha
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].plot(filtered_data['adhesivity'], filtered_data['product'])
#ax[1].plot(filtered_data2['adhesivity'], filtered_data2['product'])
#plt.xlabel('Alpha')
#plt.ylabel('Product')
#plt.title(f'Product vs Alpha for Particle Size {particle_size_value}')
#plt.show()