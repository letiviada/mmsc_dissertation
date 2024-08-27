import numpy as np
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
import pandas as pd
from utils_r import open_data_model

inputs, outputs = open_data_model('train', 'termination_time', 'regression/models_gradient_boosting')

import matplotlib.pyplot as plt

plt.hist(outputs, bins=10)
plt.xlabel('Output')
plt.ylabel('Frequency')
plt.title('Histogram of Outputs')
plt.show()