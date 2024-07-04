from utils import obtain_data
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import create_fig, style_and_colormap, save_figure

data = obtain_data('termination_time')
print(data.head())
print(data.describe())


# Plot data
# ------------
sns.set_theme()
fig, ax = create_fig(1,1,dpi = 100)
_, colors = style_and_colormap(num_positions = 7, colormap = 'tab20b')
colors = colors.tolist()
sns.scatterplot(data, x = 'alpha', y = 'termination_time', ax = ax[0],size = 'beta',legend='full', sizes = (150,150),
                 hue = 'beta', palette = colors)
ax[0].set_xlabel('Alpha',fontsize = 20)
ax[0].set_ylabel('Termination Time', fontsize = 20)
ax[0].tick_params(axis = 'both', labelsize = 20)
#ax[0].legend(title = 'Beta', bbox_to_anchor = (1,0.95), loc = 'upper left', ncols = 1)
save_figure(fig, 'figures/data_larger/termination_time_vs_alpha_beta')
