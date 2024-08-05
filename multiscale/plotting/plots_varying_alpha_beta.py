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
    if save:
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
    _, colors = style_and_colormap(num_positions=1, colormap='tab20b')
    colors = colors.tolist()
    fig, ax = create_fig(nrows = 1, ncols = 1, dpi = 100)
    ax[0].plot(data[input],data[output], linewidth = 5, color = colors[0])
    ax[0].set_xlabel(r'$\beta$')
    ax[0].set_ylabel(r'$\tau$')
    sns.despine()
    if save:
        filepath = f'regression/figures/ch3/{output}/{input}_plot'
        save_figure(fig, filepath)
    return fig, ax

