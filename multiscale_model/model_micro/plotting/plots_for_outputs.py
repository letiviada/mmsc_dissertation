import numpy as np 
import matplotlib.pyplot as plt
from .create import create_fig
from .style import style_and_colormap

def plot_time(t_eval,x_eval,funct,title = None,colormap = 'tab20',style = 'seaborn-v0_8'):
    """
    Plot for evaluating a function that depends on time and space, it has x-axis for space, y-axis for function and shows different times
    with an arrow pointing in the direction of time.

    Parameters:
    t_eval (np.ndarray): Array for the time points of size (nt,)
    x_eval (np.ndarray): Array for the spatial points of size (nx,)
    funct (n.ndarray): Array for the function of size (nt,nx)

    Returns:
    fig (matplotlib.figure.Figure): Figure for the function 
    """
    _, colors = style_and_colormap(style = style, colormap=colormap, num_positions = 1)
    nrows, ncols = 1,1
    fig, ax = create_fig(nrows, ncols, title = title)
    for t_idx,t in enumerate(t_eval):
        color = colors[t_idx % len(colors)]
        ax[0].plot(x_eval,funct[t_idx,:],color = color, linewidth = 2, label=f't = {t}')
    ax[0].set_xlabel('$x$')
    ax[0].grid(True)# Set limits to restrict the left domain to start at x=0, y=0
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)
    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.7, wspace=0.2)
    return fig

def plot_one_dim(x_eval,functs,title = None, colormap = 'tab20',style = 'seaborn-v0_8'):
    """
    Plot for evaluating a function depending on only one variable

    Parameters:
    x_eval (np.ndarray): Array for the spatial points of size (nx,)
    functs (list): List of arrays for the functions of size (nx,)

    Returns:
    fig (matplotlib.figure.Figure): Figure for the function 
    """
    _, colors = style_and_colormap(style = style, colormap=colormap, num_positions = len(functs))
    nrows, ncols = 1,1
    fig, ax = create_fig(nrows, ncols, title = title)
    for i in range(len(functs)):
        color = colors[i]
        funct = functs[i]
        ax[0].plot(x_eval,funct,color = color, linewidth = 2)
    ax[0].set_xlabel('$ \\tau$')
    ax[0].grid(True)# Set limits to restrict the left domain to start at x=0, y=0
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)
    fig.tight_layout()
    return fig