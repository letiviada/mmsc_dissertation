import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import create_fig, style_and_colormap, save_figure
import matplotlib.pyplot as plt
import numpy as np

def get_plots_size_sample(metric):
    summary_stats_poly = pd.read_csv(f'regression/sample_size_study/{metric}/summary_statistics_polynomial.csv')
    summary_stats_gb = pd.read_csv(f'regression/sample_size_study/{metric}/summary_statistics_gradient_boosting.csv')
    summary_stats_rf = pd.read_csv(f'regression/sample_size_study/{metric}/summary_statistics_random_forest.csv')


    _, colors = style_and_colormap(num_positions=3)
    fig, ax = create_fig(1,1, dpi = 100)

    ax[0].scatter(summary_stats_poly['Sample Size'], summary_stats_poly['Mean R2'], color = colors[0], marker = 'x', label = 'Polynomial')
    ax[0].scatter(summary_stats_gb['Sample Size'], summary_stats_gb['Mean R2'], color = colors[1], marker = 'x', label = 'Gradient Boosting')
    ax[0].scatter(summary_stats_rf['Sample Size'], summary_stats_rf['Mean R2'], color = colors[2], marker = 'x', label = 'Random Forest')
    ax[0].set_xlabel('Sample Size')
    ax[0].set_ylabel('Mean R2 Score')
    #ax[0].set_xlim(38.5, 181)
    ax[0].set_yticks(np.arange(0, 1.05, 0.05)) 
    ax[0].set_xticks(np.arange(30, 181, 10))
    ax[0].set_ylim(0,1.05)
    ax[0].legend(loc = 'lower right')
    plt.tight_layout()
    save_figure(fig, f'regression/figures/sample_size/comparison_{metric}')
    plt.show()

get_plots_size_sample('termination_time')
get_plots_size_sample('efficiency')
get_plots_size_sample('lifetime')