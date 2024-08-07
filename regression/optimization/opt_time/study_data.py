import pandas as pd
import sys
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import opt_ml, make_loglog, plot_optimum

time = 400
particle_sizes = [0.02, 0.04, 0.06, 0.08, 0.1]
particle_sizes = 'all'
data = pd.read_csv(f'regression/optimization/opt_time/data/time_{time}/data_for_sums.csv')
# Clean the data
cleaned_data = data[(data['adhesivity'].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])) & (data['particle_size'].isin([0.01,0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]))]

# Perform the optimization and plotting
#opt_ml(cleaned_data, f'volume_time_{time}', actual=True, predictions=False, lines=True, data_line=None, type_data='standard', particle_sizes=particle_sizes, save=True)
opt_ml(cleaned_data, f'efficiency_time_{time}', actual=True, predictions=False, lines=False, data_line=None, type_data='standard', particle_sizes=particle_sizes, save=True)
#make_loglog(cleaned_data, f'volume_time_{time}', betas=particle_sizes, type_data='standard')
#opt_ml(data,f'volume_time_{time}', actual = True, predictions = False,lines = True, data_line = None,type_data = 'standard',  particle_sizes= particle_sizes, save= False) 
#opt_ml(data,f'efficiency_time_{time}', actual = True,  predictions = False,lines = False, data_line = None,type_data = 'standard',particle_sizes= particle_sizes, save= False)
#make_loglog(data,f'volume_time_{time}', betas = particle_sizes,type_data='standard')