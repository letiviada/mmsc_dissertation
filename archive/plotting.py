import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# Labels for the non-zero entries
values_dict = {
    (1, 0, 2): "$G_{13}^{(0,1)^{T}}$",
    (1, 1, 3): "$G_{24}^{(0,1)^{T}}$",
    (3, 0, 1): "$G_{12}^{(-1,0)^{T}}$",
    (3, 2, 3): "$G_{34}^{(-1,0)^{T}}$",
    (4, 0, 1): "$G_{12}^{(0,0)^{T}}$",
    (4, 0, 2): "$G_{13}^{(0,0)^{T}}$",
    (4, 1, 0): "$G_{21}^{(0,0)^{T}}$",
    (4, 1, 3): "$G_{24}^{(0,0)^{T}}$",
    (4, 2, 0): "$G_{31}^{(0,0)^{T}}$",
    (4, 2, 3): "$G_{34}^{(0,0)^{T}}$",
    (4, 3, 1): "$G_{42}^{(0,0)^{T}}$",
    (4, 3, 2): "$G_{43}^{(0,0)^{T}}$",
    (5, 1, 0): "$G_{21}^{(1,0)^{T}}$",
    (5, 3, 2): "$G_{43}^{(1,0)^{T}}$",
    (7, 2, 0): "$G_{31}^{(0,-1)^{T}}$",
    (7, 3, 1): "$G_{42}^{(0,-1)^{T}}$"
}

def create_fig(nrows, ncols, figsize=(15, 8), dpi=300):
    """
    Create a figure and axes.

    Parameters:
    nrows (int): Number of rows of subplots.
    ncols (int): Number of columns of subplots.
    figsize (tuple): Size of the figure.
    dpi (int): Dots per inch of the figure.

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    axes (np.ndarray): Array of Axes objects.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    return fig, axes.flatten()

def save_figure(fig, path):
    """
    Saves the figure to the specified path in both SVG and PDF formats.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        path (str): The directory path where the figure should be saved.
    """
    fig.savefig(f"{path}.svg")
    fig.savefig(f"{path}.pdf")

def plot_tensor(ax, x_eval, X , tensor, t, positions, labels):
    """
    Plot tensor values for given positions and labels.
    
    Parameters:
    ax (Axes): Matplotlib axes object.
    x_eval (np.ndarray): Spatial evaluation points.
    X (np.ndarray): Numerical solution tensor.
    tensor (np.ndarray): Exact solution tensor.
    t (int): Time index.
    positions (list): List of tensor positions to plot.
    labels (list): List of labels for the plots.
    """
    num_positions = len(values_dict)
    color_map = plt.get_cmap('tab20', num_positions)
    colors = color_map(np.linspace(0, 1, num_positions))
    for pos, label,color in zip(positions, labels,colors):
        r, i, j = pos
        ax.plot(x_eval, X[t, :, r, i, j], color='k', linewidth=2, label=f"{label}")
        ax.plot(x_eval, tensor[:, r, i, j], linestyle='--', color = color, linewidth=2, label=f"Exact {label}")
    ax.set_xlabel('x')
    ax.set_title(f't={t}', pad=2)
    ax.grid(True)

def plot_results(t_eval, x_eval, X, initial_func,save = False,fig_name='conductance'):
    """
    Plot the results for all time steps.
    
    Parameters:
    t_eval (np.ndarray): Time evaluation points.
    x_eval (np.ndarray): Spatial evaluation points.
    X (np.ndarray): Numerical solution tensor.
    initial_func (function): Function to generate initial conditions.
    save_fig (bool): Flag to save the figure.
    fig_name (str): Filename for saving the figure.
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(nrows=(len(t_eval) + 2) // 3, ncols=3, figsize=(15, 8), dpi=300)
    axes = axes.flatten()

    num_positions = len(values_dict)
    color_map = plt.get_cmap('tab20', num_positions)
    colors = color_map(np.linspace(0, 1, num_positions))
    
    for idx, t in enumerate(t_eval.astype(int)):
        tensor, _ = initial_func(x_eval, t=t)
        plot_tensor(axes[idx], x_eval, X, tensor, t, list(values_dict.keys()), list(values_dict.values()))

    #handles, labels = axes[0].get_legend_handles_labels()
    #unique_labels = {label: handle for handle, label in zip(handles, labels)}
    #fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1), ncol=12)
   # Create custom legend handles
    custom_lines = []
    for color in colors:
        custom_lines.append((Line2D([0], [0], color='k', linewidth=2),
                             Line2D([0], [0], color=color, linestyle='--', linewidth=2)))
    
    custom_labels = [f"{label}" for label in values_dict.values()]

    # Creating a combined legend
    fig.legend(handles=custom_lines, labels=custom_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                loc='outside center right', borderaxespad=1, ncol=1)
    fig.suptitle('Exact (color) vs Numerical (black) Solution')
    
    if save == True:
        fig.savefig(f'tensor_code/space_dep/figures/{fig_name}.png')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75, wspace=0.4)
    #plt.show()
