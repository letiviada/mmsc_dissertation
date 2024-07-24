
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
from models.model_train import  train_model
import pandas as pd 


data = pd.read_csv('regression/optimization/optimum_values.csv')
output = 'adhesivity'
train_model(output, data, size_train = 'all', type_model = 'gradient_boosting', save = True)
