import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import open_model, change_name_time, open_data_model
from utils import get_data_from_json, data_time, obtain_data, get_ratio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np


output_name = 'volume_liquid_time_400'
time = 400
n = 1

# Machine Learnig Model
model_volume = open_model(output = 'volume_liquid_time_400', model_path = f'regression/models_polynomial/')
inputs, outputs_volume = open_data_model('total', 'volume_liquid_time_400', model_path = f'regression/models_polynomial/')
volume_predictions = model_volume.predict(inputs)
model_concentration = open_model(output = 'total_concentration_time_400', model_path=f'regression/models_polynomial/')
concentration_predictions = model_concentration.predict(inputs)
ratio = (volume_predictions ** n)/(concentration_predictions)

# Physical Model
physical_data = get_data_from_json('performance_indicators/performance_indicators_standard.json')
inputs, outputs_concentration = open_data_model('total', 'total_concentration_time_400', model_path = f'regression/models_polynomial/')
data = data_time(400, ['volume_liquid', 'total_concentration'], physical_data)
data_model = obtain_data(['volume_liquid_time_400', 'total_concentration_time_400'], data)
data_ratio = get_ratio('volume_liquid_time_400', 'total_concentration_time_400',n, data_model)

# Add predictions to the data
data_ratio['volume_predictions'] = volume_predictions
data_ratio['concentration_predictions'] = concentration_predictions
data_ratio['ratio_predictions'] = ratio


# Plot solutions
fig, axs = plt.subplots(1, 3, figsize=(10, 6))

axs[0].scatter(inputs['adhesivity'], outputs_volume, c=inputs['particle_size'], cmap='viridis')
axs[0].scatter(inputs['adhesivity'], volume_predictions, c=inputs['particle_size'], marker='x', cmap='viridis')
axs[0].set_xlabel('Adhesivity')
axs[0].set_ylabel('Outputs')
axs[0].set_title('Volume Liquid')

axs[1].scatter(inputs['adhesivity'], outputs_concentration, c=inputs['particle_size'], cmap='viridis')
axs[1].scatter(inputs['adhesivity'], concentration_predictions, c=inputs['particle_size'], marker='x', cmap='viridis')
axs[1].set_xlabel('Adhesivity')
axs[1].set_ylabel('Outputs')
axs[1].set_title('Last Concentration')

axs[2].scatter(inputs['adhesivity'], data_ratio['ratio'], c=inputs['particle_size'], cmap='viridis')
axs[2].scatter(inputs['adhesivity'], ratio, c=inputs['particle_size'], marker='x', cmap='viridis')
axs[2].set_xlabel('Adhesivity')
axs[2].set_ylabel('Ratio')
axs[2].set_title('Ratio')

plt.tight_layout()
plt.show()
