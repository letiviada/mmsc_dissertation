# Import necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Append custom library paths to system path
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')

# Import custom plotting functions
from multiscale.plotting import get_plots_size_sample, save_figure

# Generate a plot for 'efficiency' without saving it
fig, ax = get_plots_size_sample('efficiency', save=False)

# Display the plot
plt.show()