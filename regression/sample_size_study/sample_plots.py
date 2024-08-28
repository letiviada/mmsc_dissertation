import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import get_plots_size_sample, save_figure
import matplotlib.pyplot as plt

fig, ax = get_plots_size_sample('efficiency', save = False)
plt.show()