import numpy as np 
import matplotlib.pyplot as plt
from .create import create_fig
from .style import style_and_colormap, create_legend_exactandnum, create_legend_times
#rom data.helpers import map_indices

def exact_vs_num(ax, x_eval, X , tensor, t, positions, labels, colors):
    """
    Plot tensor values for given positions and labels.
    
    Parameters:
    ax (Axes): Matplotlib axes object.
    x_eval (np.ndarray): Spatial (Time) evaluation points
    X (np.ndarray): Numerical solution tensor.
    tensor (np.ndarray): Exact solution tensor.
    t (int): Time index.
    positions (list): List of tensor positions to plot.
    labels (list): List of labels for the plots.
    colors (list): List of colours generated from the colourmap

    """
    for pos, label,color in zip(positions, labels,colors):
        i, j,r,s = pos
        ax.plot(x_eval, tensor[:, i, j,r,s], color = 'k', linewidth=2, label=f"Exact {label}")
        ax.plot(x_eval, X[t, :, i, j,r,s], linestyle= '--', color=color, linewidth=2, label=f"{label}")
        
    ax.set_xlabel('$x$')
    ax.set_title(f'$t={t}$', pad=2)
    ax.grid(True)
    

def specific_cell(ax,t_eval,x_eval,cell,num_approx,exact_sol, colors):
    """"
    Plot the results for a specific edge conductance and varying time. [Add arrow representing direction of time]

    Parameters:
    ax (Axes): Matplotlib axes object
    t_eval (np.ndarray): Time evaluation points
    x_eval (np.ndarray): Spatial evaluation points.
    cell (tuple): Edge evaluated (i,j,r,s)
    num_approx (np.ndarray): Numerical solution tensor of shape (nt,nx,i,j,r,s).
    exact_sol (function): Function to generate the exact solution for a specific time
    colors (list): List of colours generated from the colourmap

    Returns:
    fig (matplotlib.figure.Figure): The created figure.

    """
    i,j,r,s = cell
    ax.set_title(f"$G_{{{i}{j}}}^{{({r},{s})^T}}$",pad = 2)
    cell = 1
    i,j,r,s = cell
    for t_idx,t in enumerate(t_eval):
        tensor, _ = exact_sol(x_eval, shape = (4,4,3,3), t=t)
        color = colors[t_idx % len(colors)]
        ax.plot(x_eval,tensor[:,i,j,r,s], color='k', linewidth = 2, label=f'Exact t = {t}')
        ax.plot(x_eval,num_approx[t_idx,:,i,j,r,s],color=color,linestyle='--', linewidth = 2, label=f't = {t}')
    ax.set_xlabel('$x$')
    ax.grid(True)# Set limits to restrict the left domain to start at x=0, y=0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
     # Determine the center and a point near the origin
    x_center = x_eval[len(x_eval) // 2]
    y_center = (num_approx[:, :, i, j, r, s].max() + num_approx[:, :, i, j, r, s].min()) / 2
    x_start = 0.05 * x_eval[-1]
    y_start = 0.05 * num_approx[:, :, i, j, r, s].max()

    # Add an arrow to show the direction of time from the center to near the origin, closer to the largest time curve
    ax.annotate('', xy=(x_start, y_start), xytext=(x_center, y_center),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
    # Add text "Time" at the end of the arrow
    ax.text(x_center, y_center, 'Time', fontsize=12, verticalalignment='bottom')


def plot_varying_cell(t_eval,x_eval,num_approx,exact_sol,cells,colormap= 'tab20', style= 'seaborn-v0_8'):
    """
    Plot the results for all time steps with color handling.
    
    Parameters:
    t_eval (np.ndarray): Time evaluation points.
    x_eval (np.ndarray): Spatial evaluation points.
    num_approx (np.ndarray): Numerical solution tensor of shape (nt,nx,i,j,r,s).
    exact_sol (function): Function to generate exact solution for a specific time.
    cells (list): list of cells we want to plot.
    colormap (str): Colormap to use for the plots.
    style (str): Style of the figure

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    """
    _, colors = style_and_colormap(style = style, colormap=colormap, num_positions=len(t_eval))
    nrows, ncols = (len(cells) + 2) // 4, min(len(cells),4)
    fig, axes = create_fig(nrows, ncols, title = 'Exact (black) vs Numerical (color) Solution',figsize=(20,10))
    for idx,cell in enumerate(cells):
        specific_cell(axes[idx], t_eval, x_eval, cell, num_approx, exact_sol, colors)

    create_legend_times(fig, t_eval, style = style, colormap = colormap)
    fig.tight_layout(rect=[0,0,0.93,1.0])
    #fig.subplots_adjust(hspace=0.7, wspace=0.2)

    return fig



def plot_varying_time(t_eval,x_eval,num_approx,exact_sol,values_dict, colormap = 'tab20',style = 'seaborn-v0_8'):
    """
    Plot the results for all time steps with color handling.
    
    Parameters:
    t_eval (np.ndarray): Time evaluation points.
    x_eval (np.ndarray): Spatial evaluation points.
    num_approx (np.ndarray): Numerical solution tensor of shape (nt,nx,i,j,r,s).
    exact_sol (function): Function to generate exact solution for a specific time.
    colormap (str): Colormap to use for the plots.
    style (str): Style of the figure

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    """
    _, colors = style_and_colormap(style = style, colormap=colormap, num_positions=len(values_dict))
    nrows, ncols = (len(t_eval) + 2) // 3, min(len(t_eval),3)
    fig, axes = create_fig(nrows, ncols, title = 'Exact (black) vs Numerical (color) Solution')
    for idx, t in enumerate(t_eval.astype(int)):
        tensor, _ = exact_sol(x_eval, shape = (4,4,3,3), t=t)
        exact_vs_num(axes[idx], x_eval, num_approx, tensor, t, list(values_dict.keys()), list(values_dict.values()), colors)

    create_legend_exactandnum(fig, values_dict, style = style, colormap = colormap)
    plt.subplots_adjust(hspace=0.27, wspace=0.27)

    return fig

