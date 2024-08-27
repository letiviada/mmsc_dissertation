import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import get_plots_size_sample, save_figure
import matplotlib.pyplot as plt

fig, ax = get_plots_size_sample('total_throughput', save = False)
ax[0].set_ylabel(r'$\bar{R}^2$')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "font.size": 50
    #"axes.labelsize": 36,
    #"axes.titlesize": 36,
    #"legend.fontsize": 36,
    #"xtick.labelsize": 36,
    #"ytick.labelsize": 36
    })
ax[0].set_xlabel('Sample size')
ax[0].set_xticks([40,60,80,100,120,140,160,180])
ax[0].set_ylim(bottom=0.8, top = 1.01)
plt.show()
save_figure(fig, 'regression/figures/sample_size_study/total_throughput/total_throughput_GB')
#get_plots_size_sample('efficiency')
#get_plots_size_sample('lifetime')3