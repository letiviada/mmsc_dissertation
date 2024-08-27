import matplotlib.pyplot as plt
import seaborn as sns
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import pandas as pd
import numpy as np

def plot_time_opt(data:pd.DataFrame, particle_size: float, weight_coefficient: list, data_ml: pd.DataFrame = None, save = True)->tuple:
    _, colors = style_and_colormap(num_positions = len(weight_coefficient), colormap = 'tab10')
    colors = colors.tolist()
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    data_plot = data[data['particle_size'] == particle_size]

    if data_ml is not None:
        data_plot_ml = data_ml[data_ml['particle_size'] == particle_size]
        print(data_plot_ml)
    for weight in weight_coefficient:
        data_weight = data_plot[data_plot['weight_coefficient'] == weight]
        ax[0].scatter(data_weight['adhesivity'], data_weight['gamma'], 
                   color = colors[weight_coefficient.index(weight)], s = 300, label = f'Weight = {weight}')
        if data_ml is not None:
            data_weight_ml = data_plot_ml[data_plot_ml['weight_coefficient'] == weight]
            ax[0].plot(data_weight_ml['adhesivity'], data_weight_ml['gamma'], 
                       color = colors[weight_coefficient.index(weight)], linewidth = 5)
    ax[0].set_xticks(np.arange(0,1.1,0.2))
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\gamma$')
    if save:
        save_figure(fig, 'regression/optimization/opt_throughput/plots/throughput_100/plot_varying_weight_large')
    return fig, ax


def plot_one_weight(data:pd.DataFrame, weight_coefficient: float, particle_size: list, data_ml: pd.DataFrame = None, save = True)->tuple:
    _, colors = style_and_colormap(num_positions = len(particle_size), colormap = 'tab10')
    colors = colors.tolist()
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    data_plot = data[data['weight_coefficient'] == weight_coefficient]

    if data_ml is not None:
        data_plot_ml = data_ml[data_ml['weight_coefficient'] == weight_coefficient]
    for part_size in particle_size:
        data_part_size = data_plot[data_plot['particle_size'] == part_size]
        ax[0].scatter(data_part_size['adhesivity'], data_part_size['gamma'], 
                   color = colors[particle_size.index(part_size)],  label = f'PS = {part_size}', s = 300)
        if data_ml is not None:
            data_weight_ml = data_plot_ml[data_plot_ml['particle_size'] == part_size]
            ax[0].plot(data_weight_ml['adhesivity'], data_weight_ml['gamma'], 
                       color = colors[particle_size.index(part_size)], linewidth = 5)
    ax[0].set_xticks(np.arange(0,1.1,0.2))
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\gamma$')
    if save == True:
        save_figure(fig, 'regression/optimization/opt_throughput/plots/throughput_100/plot_varying_beta')
    return fig, ax

def plot_for_spec_throughput(data:pd.DataFrame, output:str, particle_size,data_marker:pd.DataFrame = None, save:bool = True):
    if type(particle_size)==str and particle_size == 'all':
        particle_size = data['particle_size'].unique().tolist()
    _, colors = style_and_colormap(num_positions = len(particle_size), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(particle_size, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    for part_size in particle_size:
        data_part_size = data[data['particle_size'] == part_size]
        data_marker_ps = data_marker[data_marker['particle_size'] == part_size]
        ax[0].plot(data_part_size['adhesivity'], data_part_size[output], 
                   color = color_mapping[part_size],  label = f'PS = {part_size}', linewidth = 5)
        ax[0].scatter(data_marker_ps['adhesivity'], data_marker_ps[output],
                      color = color_mapping[part_size], s = 300)
        
    ax[0].set_xticks(np.arange(0,1.1,0.2))
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\eta(\theta = 100)$')
    if save == True:
        save_figure(fig, 'regression/optimization/opt_throughput/plots/throughput_100/plot_varying_alpha')
       # save_figure(fig, 'regression/optimization/opt_time/plots/time_400/efficiency_beta_even')
    return fig, ax

def plot_for_varying_beta(data:pd.DataFrame, output:str, adhesivity,data_marker:pd.DataFrame = None, save:bool = True):
    if type(adhesivity)==str and adhesivity == 'all':
        adhesivity = data['adhesivity'].unique().tolist()
    _, colors = style_and_colormap(num_positions = len(adhesivity), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(adhesivity, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    for adherence in adhesivity:
        data_part_size = data[data['adhesivity'] == adherence]
        data_marker_ps = data_marker[data_marker['adhesivity'] == adherence]
        ax[0].plot(data_part_size['particle_size'], data_part_size[output], 
                   color = color_mapping[adherence],  label = f'PS = {adherence}', linewidth = 5)
        ax[0].scatter(data_marker_ps['particle_size'], data_marker_ps[output],
                      color = color_mapping[adherence], s = 300)
        
    ax[0].set_xticks(np.arange(0.01,0.11,0.02))
    ax[0].set_xlabel(r'$\beta$')
    ax[0].set_ylabel(r'$\eta(\theta = 100)$')
    ax[0].set_ylim(top= 1.0)
    if save == True:
        save_figure(fig, 'regression/optimization/opt_throughput/plots/throughput_100/efficiency_varying_beta')
        #save_figure(fig, 'regression/optimization/opt_time/plots/time_400/efficiency_alpha_even')
    return fig, ax

def plot_range_time(data:pd.DataFrame,time:float, particle_size: list, data_ml:pd.DataFrame, save:bool=True):
    if type(particle_size) == str and particle_size == 'all':
        particle_size = data['particle_size'].unique().tolist()
    _, colors = style_and_colormap(num_positions = len(particle_size), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(particle_size, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,15), dpi = 100)
    for part_size in particle_size:
        data_part_size = data[data['particle_size'] == part_size]
        sns.lineplot(data = data_part_size, x = 'weight_coefficient', y = f'adhesivity_time_{time}',ax = ax[0], 
                   color = color_mapping[part_size],  label = f'PS = {part_size}', linewidth = 5)

        if data_ml is not None:
            data_part_size_ml = data_ml[data_ml['particle_size'] == part_size]
            sns.lineplot(data = data_part_size_ml, x = 'weight_coefficient', y = f'adhesivity_time_{time}',ax = ax[0], 
                   color = color_mapping[part_size], linestyle = '--', label = f'PS = {part_size}', linewidth = 5)
    ax[0].set_xlabel(r'$n$')
    ax[0].set_ylabel(r'$\alpha_{\mathrm{max}}$')
    if save == True:
        save_figure(fig, 'regression/optimization/opt_time/plots/time_400/alpha_max_range_large')

    return fig, ax
def plot_range_throughput(data:pd.DataFrame,throughput:float, particle_size: list,data_ml:pd.DataFrame, save:bool=True):
    if type(particle_size) == str and particle_size == 'all':
        particle_size = data['particle_size'].unique().tolist()
    _, colors = style_and_colormap(num_positions = len(particle_size), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(particle_size, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    for part_size in particle_size:
        data_part_size = data[data['particle_size'] == part_size]
        sns.lineplot(data = data_part_size, x = 'weight_coefficient', y = f'adhesivity_throughput_{throughput}',ax = ax[0], 
                   color = color_mapping[part_size],  label = f'PS = {part_size}', linewidth = 5)
        if data_ml is not None:
            data_part_size_ml = data_ml[data_ml['particle_size'] == part_size]
            sns.lineplot(data = data_part_size_ml, x = 'weight_coefficient', y = f'adhesivity_throughput_{throughput}',ax = ax[0], 
                   color = color_mapping[part_size], linestyle = '--', label = f'PS = {part_size}', linewidth = 5)
    ax[0].set_xlabel(r'$n$')
    ax[0].set_ylabel(r'$\alpha_{\mathrm{max}}$')
    if save == True:
        save_figure(fig, 'regression/optimization/opt_throughput/plots/throughput_100/alphamax_range_sm')

    return fig, ax

def pareto_front(data:pd.DataFrame, xaxis:str, yaxis:str, min_throughput_value:float, min_efficiency_value:float, adhesivity:list, save:bool):
    if type(adhesivity)==str and adhesivity == 'all':
        adhesivity = data['particle_size'].unique().tolist()
    _, colors = style_and_colormap(num_positions = len(adhesivity), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(adhesivity, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)

    for adherence in adhesivity:
        data_part_size = data[data['particle_size'] == adherence]
        ax[0].plot(data_part_size[xaxis], data_part_size[yaxis], color = color_mapping[adherence], linewidth = 5)
        min_throughput = data_part_size[data_part_size[xaxis] <= min_throughput_value]
        min_efficiency = data_part_size[data_part_size[yaxis] >= min_efficiency_value]
        min_throughput_row = min_throughput.loc[min_throughput[xaxis].idxmax()]
        min_efficiency_row = min_efficiency.loc[min_efficiency[yaxis].idxmin()]
        print(min_throughput_row)
        print(min_efficiency_row)
        ax[0].scatter(min_throughput_row[xaxis], min_throughput_row[yaxis], color = color_mapping[adherence],marker = 'X', s = 300, zorder = 5)
        ax[0].scatter(min_efficiency_row[xaxis], min_efficiency_row[yaxis], color = color_mapping[adherence],marker = 'X', s = 300, zorder = 5)
    ax[0].axvline(min_throughput_value, color='green', linestyle='-.',linewidth = 3,  label='Min Throughput')
    ax[0].axhline(min_efficiency_value, color='green', linestyle='-.',linewidth = 3,  label='Min Efficiency')
    ax[0].set_xlabel(r'$\theta(\min\{\tau,400\})$')
    ax[0].set_ylabel(r'$\eta(\min\{\tau,400\})$')
    ax[0].text(300, min_efficiency_value + 0.02, r'$\eta_{\mathrm{min}}$', color='black', ha='center')
    ax[0].text(min_throughput_value + 10, 1, r'$T_{\mathrm{max}}$', color='black', ha='center')
    ax[0].fill_betweenx(y=np.linspace(min_efficiency_value, 1.0, 100),
                  x1=min_throughput_value, x2=100,
                  color='green', alpha=0.1, label='Region of Interest')

    
    if save == True:
        save_figure(fig, 'regression/optimization/opt_time/plots/time_400/pareto_front')
    return fig, ax