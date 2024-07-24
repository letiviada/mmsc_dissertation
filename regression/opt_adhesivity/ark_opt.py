import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import clean_data
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/dissertation_mmsc')
from multiscale.plotting import plot_adhesivity
import pandas as pd
import numpy as np

#filename = 'performance_indicators/performance_indicators_standard_basic.json'
#data = clean_data(filename)
#plot_adhesivity(data, output = 'avg_removed_particles',particle_sizes='all', save = True)


n = 1.69
filename = f'regression/optimization/data_varying_n.csv'
df = pd.read_csv(filename)
print(df.head())
new_df = df[df['n'] == n]

plot_adhesivity(new_df, output = 'product',particle_sizes='all', save = False)