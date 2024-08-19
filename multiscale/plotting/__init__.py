from .create import create_fig
from .plots_for_outputs import plot_time, plot_one_dim, pressure_plot
from .save import save_figure
from .style import style_and_colormap
from .plots_data_ml import scatter_histogram, boxplot, view_data_all_outputs,scatter_solutions, plot_optimum_models
from .plots_data_ml import opt_ml, model_plot_with_lines_and_scatter, make_loglog, plot_optimal_adhesivity, get_plots_size_sample, plot_optimum
from .plots_varying_alpha_beta import plot_adhesivity, plot_perf_ind, plot_perf_ind_various_bet, plot_perf_ind_time
from .plots_optimization import plot_time_opt, plot_one_weight, plot_for_spec_throughput, plot_for_varying_beta, plot_range_time,  pareto_front