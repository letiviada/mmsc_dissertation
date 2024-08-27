# This file contains the functions for plotting the outputs of the model
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import numpy as np
import seaborn as sns

def plot_time(t_eval,x_eval,funct,title = None,colormap = 'tab10',style = 'seaborn-v0_8'):
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
    _, colors = style_and_colormap( colormap=colormap, num_positions = len(t_eval))
    nrows, ncols = 1,1
    fig, ax = create_fig(nrows, ncols,figsize=(15,8), title = title)
    for t_idx,t in enumerate(t_eval):
        color = colors[t_idx]
        ax[0].plot(x_eval,funct[t_idx,:],color = color, linewidth = 5, label=f't = {t:.2f}')
    ax[0].set_xticks(np.arange(0,1.1,0.2))
    ax[0].set_yticks(np.arange(0,1.3,0.2))
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)   
    fig.tight_layout()
    sns.despine()
    #fig.subplots_adjust(hspace=0.7, wspace=0.2)
    return fig, ax

def plot_one_dim(x_eval,functs,title = None, colormap = 'tab10',style = 'seaborn-v0_8', funct_type = None):
    """
    Plot for evaluating a function depending on only one variable

    Parameters:
    x_eval (np.ndarray): Array for the spatial points of size (nx,)
    functs (list): List of arrays for the functions of size (nx,)

    Returns:
    fig (matplotlib.figure.Figure): Figure for the function 
    """
    _, colors = style_and_colormap( colormap=colormap, num_positions = len(functs))
    nrows, ncols = 1,1
    fig, ax = create_fig(nrows, ncols,figsize=(15,8), title = title)
    for i in range(len(functs)):
        color = colors[i]
        funct = functs[i]
        if funct_type == 'velocity':
            funct_array = np.array(funct)
            x_eval_array = np.array(x_eval)
            filtered_idx = funct_array >= 0.1
            filtered_y = funct_array[filtered_idx]
            filtered_x = x_eval_array[filtered_idx]
            ax[0].plot(filtered_x,filtered_y,color = color, linewidth = 5, label = i)
        else:
            ax[0].plot(x_eval,funct,color = color, linewidth = 5, label= i)
    if funct_type == 'velocity':
        ax[0].axhline(y=0.1,color='orange',linestyle='--',linewidth=5)
# Set limits to restrict the left domain to start at x=0, y=0
    ax[0].set_xlim(left=0)
    fig.tight_layout()
    sns.despine()
    return fig, ax

def pressure_plot(p:np.ndarray, dpdx: np.ndarray, x_eval: np.ndarray):
    """
    Pressure and pressure gradient plot or different times.

    Parameters:
    -----------
    p (np.ndarray): Array of size (nt,nx) for the pressure
    dpdx (np.ndarray): Array of size (nt,nx) for the pressure gradient
    x_eval (np.ndarray): Array of size (nx,) for the spatial points

    Returns:
    --------
    fig (matplotlib.figure.Figure): Figure for the pressure and pressure gradient
    """
    _, colors = style_and_colormap(num_positions=p.shape[0], colormap='tab10')
    nrows, ncols = 1,1 
    fig_p, ax_p = create_fig(nrows, ncols, figsize=(15,8))
    sns.despine()
    fig_dpdx, ax_dpdx = create_fig(nrows, ncols, figsize=(15,8))
    for i in range(p.shape[0]):
        color = colors[i]
        ax_p[0].plot(x_eval,p[i,:],color = color, linewidth = 5)
        ax_dpdx[0].plot(x_eval,dpdx[i,:],color = color, linewidth = 5)
    ax_p[0].set_xlabel('$x$')
    ax_p[0].set_xlim(left=0)
    ax_p[0].set_xticks(np.arange(0,1.1,0.2).round(2))
    ax_dpdx[0].set_xlim(left=0, right = 1)
    ax_p[0].set_ylim(bottom=0)
    ax_dpdx[0].set_ylim(top=0)
    ax_dpdx[0].set_xlabel('$x$')
    ax_p[0].set_ylabel('$p$')
    ax_dpdx[0].set_ylabel(r'$\partial p /\partial x$')
    sns.despine()
    save_figure(fig_p, 'multiscale/figures/mono-dispersed/macroscale/pressure/pressure')
   # save_figure(fig_dpdx, 'multiscale/figures/mono-dispersed/macroscale/pressure/pressure_grad')
    return fig_p, fig_dpdx

