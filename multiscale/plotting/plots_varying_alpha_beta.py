import matplotlib.pyplot as plt
import seaborn as sns
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import numpy as np
import pandas as pd
from scipy.stats import linregress

def plot_adhesivity(data:pd.DataFrame, output:str, particle_sizes:list, save: bool):
    """
    Plot the adhesivity of the model against the adhesivity of the data.

    Parameters:
    data (pd.DataFrame): The data to plot.
    output (str): The output to plot.
    """
    if particle_sizes == 'all':
        particle_sizes = data['particle_size'].unique()
    unique_keys = data['particle_size'].unique()
    _, colors = style_and_colormap(num_positions=len(unique_keys), colormap='tab20b')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(unique_keys, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1 ,dpi = 100)
    for particle_size in particle_sizes:
        data_particle_size = data[data['particle_size'] == particle_size]
        sns.scatterplot(x = data_particle_size['adhesivity'],y =  data_particle_size[output], ax=ax[0], color=color_mapping[particle_size],
                         label = f'Particle size: {particle_size}')
    if save==True:
        filepath = f'regression/figures/{output}/adhesivity_plot'
        save_figure(fig, filepath)
    return fig, ax

def plot_perf_ind(data:pd.DataFrame, output:str, input:str, save:bool):
    """
    Plot performance indicators against an input variable.

    Parameters:
    -----------
    data (pd.DataFrame): The data to plot.
    output (str): The output to plot.
    input (str): The input to plot.

    Returns:
    --------
    fig, ax (tuple): The figure and axis of the plot.
    """
    _, colors = style_and_colormap(num_positions=1, colormap='tab10')
    colors = colors.tolist()
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    ax[0].plot(data[input],data[output], linewidth = 5, color = colors[0])
    ax[0].set_xlabel(r'$\beta$')
    ax[0].set_ylabel(r'$\tau$')
    #ax[0].set_xlabel(r'$\alpha$')
    #ax[0].set_ylabel(r'$\eta(\tau)$')
    ax[0].set_xticks(np.arange(0.02,0.11,0.02))
    ax[0].set_yticks([250,500,750,1000,1250,1500,1750])
    sns.despine()
    if save==True:
        filepath = f'regression/figures/ch3/{output}/{input}_plot'
        save_figure(fig, filepath)
    return fig, ax

def plot_perf_ind_various_bet(data:pd.DataFrame, output:str,particle_size:list, input:str = 'adhesivity',save:bool = True):
    if type(particle_size) == str and particle_size == 'all':
        particle_size = data['particle_size'].unique()
    _, colors = style_and_colormap(num_positions = len(particle_size), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(particle_size, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    for part_size in particle_size:
        data_part_size = data[data['particle_size'] == part_size]

        ax[0].plot(data_part_size[input], data_part_size[output], 
                   color = color_mapping[part_size],  label = f'PS = {part_size}', linewidth = 5)
    ax[0].set_xticks(np.arange(0.2,1.1,0.2))
    #ax[0].set_xlabel(r'$\alpha$')
    #ax[0].set_ylabel(r'$\theta(\tau)$')
    ax[0].set_ylabel(r'$\theta(t)$')
    ax[0].set_xlabel(r'$t$')
    if save == True:
        filepath = f'regression/figures/ch3/performance_metrics/{output}/alpha_even_time'
        save_figure(fig, filepath)
    return fig, ax

def plot_perf_ind_time(data:pd.DataFrame, output:str,particle_size:list, input_value:str = 'time',save:bool = True):
    if type(particle_size) == str and particle_size == 'all':
        particle_size = data['adhesivity'].unique()
    _, colors = style_and_colormap(num_positions = len(particle_size), colormap = 'tab10')
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(particle_size, colors)}
    fig, ax = create_fig(nrows = 1, ncols = 1, figsize = (15,8), dpi = 100)
    for part_size in particle_size:
        data_part_size = data[data['adhesivity'] == part_size]
        data_part_size = data_part_size.drop(['adhesivity', 'particle_size'], axis = 1)
        print(data_part_size)
        inputs = data_part_size[input_value].values[0]
        outputs = data_part_size[output].values[0]
        ax[0].plot(inputs, outputs, 
                   color = color_mapping[part_size],  label = f'PS = {part_size}', linewidth = 5)
   # ax[0].set_xticks(np.arange(0.2,1.1,0.2))
    ax[0].set_ylabel(r'$\theta(t)$')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylim(0, 300)
    if save == True:
        filepath = f'regression/figures/ch3/performance_metrics/{output}/alpha_even_time'
        save_figure(fig, filepath)
    return fig, ax

