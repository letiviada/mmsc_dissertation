import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import create_fig, style_and_colormap, save_figure
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Import the necessary libraries

# Step 2: Read the CSV file
summary_stats = pd.read_csv('summary_statistics_by_sample_size.csv')

# Step 3: Plot the results
_, colors = style_and_colormap(num_positions=1)
fig, ax = create_fig(1,1, dpi = 100)

ax[0].scatter(summary_stats['Sample Size'], summary_stats['Mean R2'], color = colors[0], marker = 'x')
ax[0].set_xlabel('Sample Size')
ax[0].set_ylabel('Mean R2 Score')
ax[0].set_yticks(np.arange(0.5, 1.05, 0.05)) 
plt.tight_layout()
save_figure(fig, 'regression/figures/sample_size/comparison')
plt.show()
