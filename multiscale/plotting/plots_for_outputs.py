# This file contains the functions for plotting the outputs of the model
from .create import create_fig
from .style import style_and_colormap
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
    fig, ax = create_fig(nrows, ncols, title = title)
    for t_idx,t in enumerate(t_eval):
        color = colors[t_idx]
        ax[0].plot(x_eval,funct[t_idx,:],color = color, linewidth = 5, label=f't = {t:.2f}')
    # Calculate direction of growth for arrows
    mid_index = len(x_eval) // 2 
    start_point = (x_eval[mid_index], funct[0, mid_index] - 0.02 )  # t = 0 function
    end_point = (x_eval[mid_index], funct[-1, mid_index] + 0.03)  # t = final function
    ax[0].annotate('', xy=(end_point[0], end_point[1]), xytext=(start_point[0], start_point[1]),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
    # Add text "Time" at the end of the arrow
    ax[0].text(end_point[0] + 0.05, end_point[1], 'Time', fontsize=18, verticalalignment='bottom')    
    ax[0].set_xlabel('$x$')
# Set limits to restrict the left domain to start at x=0, y=0
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)
    ax[0].legend( loc = 'center left',bbox_to_anchor= (1,0.5))
    fig.tight_layout()
    sns.despine()
    #fig.subplots_adjust(hspace=0.7, wspace=0.2)
    return fig

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
    fig, ax = create_fig(nrows, ncols, title = title)
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
    return fig
