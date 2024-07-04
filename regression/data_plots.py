from utils import obtain_data
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
#sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import create_fig, style_and_colormap, save_figure

data = obtain_data('termination_time', 
filename = '/Users/letiviada/dissertation_mmsc/multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json')


# Plot data
# ------------
_, colors = style_and_colormap(num_positions = 9, colormap = 'tab20b')#
fig, ax = create_fig(1,2,dpi = 100)
colors = colors.tolist()
sns.scatterplot(data, x='alpha', y='termination_time', ax=ax[0], size='beta', sizes=(150, 150),
                hue='beta', palette=colors[:7], legend=False),
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('Termination Time')
#ax[0].legend(title = 'Beta',bbox_to_anchor=(0.75,1.5), loc='best', ncols=4)

sns.histplot(data = data, x = 'termination_time', ax = ax[1], bins = 20, kde = True, color = colors[7])
ax[1].set_xlabel('Termination Time')
ax[1].set_ylabel('Frequency')

fig2, ax2 = create_fig(1,2,dpi = 100)
sns.boxplot(data = data, x = 'alpha', y = 'termination_time', ax = ax2[0], color = colors[7])
ax2[0].set_xlabel('Alpha')
ax2[0].set_ylabel('Termination Time')

sns.boxplot(data = data, x = 'beta', y = 'termination_time', ax = ax2[1], color = colors[7])
ax2[1].set_xlabel('Beta')
ax2[1].set_ylabel('Termination Time')
plt.tight_layout()
save_figure(fig, 'regression/figures/data/termination_time_vs_alpha_beta')
save_figure(fig2, 'regression/figures/data/boxplot_termination_time_vs_alpha_beta')
