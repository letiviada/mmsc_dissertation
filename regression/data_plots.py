from treat_data import get_data_from_json, get_input_output
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import create_fig, style_and_colormap, save_figure
# Get the data from the json file and split it into inputs and outputs
data = get_input_output(get_data_from_json(filename = 'multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json'), 'termination_time')
data = data.drop(data[(data['alpha'] == 0.3) & (data['beta'] == 0.03)].index)
inputs = data[['alpha', 'beta']]
ouputs = data['termination_time']


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
save_figure(fig, 'regression/figures/data/termination_time_vs_alpha_beta')