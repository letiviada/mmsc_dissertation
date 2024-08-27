import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/multiscale/')
from multiscale.plotting import plot_optimum, plot_optimum_models, save_figure
from regression.optimization.opt_ml_model import get_data_opt_adhesivity_plots, train_and_plot_opt_adhesivity
import matplotlib.pyplot as plt

def get_data(type_model):
    data = pd.read_csv(f'regression/optimization/opt_throughput/data/physical/optimum_values.csv')
    data_pred, model1 = get_data_opt_adhesivity_plots(data, name_model = f'throughput_100',type_model = type_model)
    model_value =  f'time_400'
    return data_pred, model_value

data_pred_rf, model_value_rf = get_data('random_forest')
data_pred_gb, model_value_gb = get_data('gradient_boosting')
data_pred_poly, model_value_poly = get_data('polynomial')

data = pd.read_csv(f'regression/optimization/opt_throughput/data/physical/optimum_values.csv')

filtered_data = data[data['particle_size'] == 0.06]
filtered_data_rf = data_pred_rf[data_pred_rf['particle_size'] == 0.06]
filtered_data_gb = data_pred_gb[data_pred_gb['particle_size'] == 0.06]
filtered_data_poly = data_pred_poly[data_pred_poly['particle_size'] == 0.06]

fig, ax, color_mapping = plot_optimum_models(['actual','random_forest','gradient_boosting','polynomial'])
ax[0].scatter(filtered_data['weight_coefficient'], filtered_data[f'adhesivity_throughput_100'], 
              color = color_mapping['actual'], s = 350, label = 'actual', alpha  = 0.7)

#ax[0].scatter(filtered_data_rf['n'], filtered_data_rf['adhesivity_predictions'], marker = 'x', color = color_mapping['random_forest'], s = 200, linewidth = 5, alpha = 0.9, label = 'random_forest')
#ax[0].scatter(filtered_data_gb['n'], filtered_data_gb['adhesivity_predictions'],
     #        marker = 'x', color = color_mapping['gradient_boosting'], s = 200, alpha = 0.9,linewidth = 5, label = 'gradient_boosting')

ax[0].scatter(filtered_data_poly['n'], filtered_data_poly['adhesivity_predictions'],
                marker = 'x', color = color_mapping['polynomial'], s = 200, alpha = 0.9, linewidth = 5, label = 'polynomial')


ax[0].set_ylabel(r'$\alpha_{\mathrm{max}}$')
ax[0].set_xlabel(r'$n$')
save_figure(fig, 'regression/optimization/opt_throughput/plots/actual_vs_poly')

plt.show()