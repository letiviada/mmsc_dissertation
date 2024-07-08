import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import create_fig, style_and_colormap
import matplotlib.pyplot as plt

# Step 1: Import the necessary libraries

# Step 2: Read the CSV file
summary_stats = pd.read_csv('summary_statistics_by_sample_size.csv')

# Step 3: Plot the results
fig, ax = create_fig(1,1)
_, colors = style_and_colormap(num_positions=1)
ax[0].scatter(summary_stats['Sample Size'], summary_stats['Mean R2'], color = colors[0])
ax[0].xlabel('Sample Size')
ax[0].ylabel('Mean R2 Score')
plt.show()
