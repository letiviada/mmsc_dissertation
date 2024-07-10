import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from utils import open_model, change_name_time

name = 'Volume Liquid'
time = 400

output_name_folder = change_name_time(name, time)
model_volume = open_model(output_name_folder)
print(model_volume)