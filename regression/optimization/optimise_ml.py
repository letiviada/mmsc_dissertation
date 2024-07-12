import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import open_model, change_name_time, open_data_model
from utils import get_data_from_json, data_time, obtain_data
import matplotlib.pyplot as plt

name = 'volume_liquid_time_400'
time = 400


# Machine Learnig Model
model_volume = open_model(name)
inputs, outputs = open_data_model('test', 'volume_liquid_time_400',
                                   'performance_indicators/performance_indicators_phi_1.0.json')
volume_predictions = model_volume.predict(inputs)

# Physical Model
physical_data = get_data_from_json('performance_indicators/performance_indicators_phi_1.0.json')
data = data_time(400, ['volume_liquid', 'last_concentration'], physical_data)
data_model = obtain_data(['volume_liquid_time_400', 'last_concentration_time_400'], data)

# Plot solutions
fig = plt.figure(figsize=(12, 8))
plt.scatter(inputs['adhesivity'], outputs, c=inputs['particle_size'], cmap='viridis')
plt.scatter(inputs['adhesivity'], volume_predictions, c=inputs['particle_size'],marker = 'x', cmap='viridis')
plt.xlabel('Adhesivity')
plt.ylabel('Particle Size')
plt.show()