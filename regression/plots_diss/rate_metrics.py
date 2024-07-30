import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/regression')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import opt_ml, make_loglog, plot_optimum
from regression.utils import get_data_from_json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = get_data_from_json('performance_indicators/performance_indicators_standard_basic.json')
data = data[['adhesivity', 'termination_time', 'particle_size', 'efficiency', 'lifetime']]
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
