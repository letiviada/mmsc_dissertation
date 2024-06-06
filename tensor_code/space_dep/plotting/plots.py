import numpy as np 
import matplotlib.pyplot as plt
from .create import create_fig
from .style import style_and_colormap, create_legend_exactandnum

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
        r, i, j = pos
        ax.plot(x_eval, X[t, :, r, i, j], color='k', linewidth=2, label=f"{label}")
        ax.plot(x_eval, tensor[:, r, i, j], linestyle='--', color = color, linewidth=2, label=f"Exact {label}")
    ax.set_xlabel('x')
    ax.set_title(f't={t}', pad=2)
    ax.grid(True)

def plot_varying_time(t_eval,x_eval,num_approx,exact_sol,values_dict, colormap = 'tab20',style = 'seaborn-v0_8'):
    """
    Plot the results for all time steps with color handling.
    
    Parameters:
    t_eval (np.ndarray): Time evaluation points.
    x_eval (np.ndarray): Spatial evaluation points.
    num_approx (np.ndarray): Numerical solution tensor of shape (nt,nx,i,j,r,s).
    exact_sol (function): Function to generate exact solution for a specific time.
    colormap (str): Colormap to use for the plots.
    title (str): Title of the figure

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    """
    _, colors = style_and_colormap(style = style, colormap=colormap, num_positions=len(values_dict))
    nrows, ncols = (len(t_eval) + 2) // 3, 3
    fig, axes = create_fig(nrows, ncols, title = 'Exact (black) vs Numerical (color) Solution')
    for idx, t in enumerate(t_eval.astype(int)):
        tensor, _ = exact_sol(x_eval, t=t)
        exact_vs_num(axes[idx], x_eval, num_approx, tensor, t, list(values_dict.keys()), list(values_dict.values()), colors)

    create_legend_exactandnum(fig, values_dict, style = style, colormap = colormap)
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.75, wspace=0.4)

    return fig


