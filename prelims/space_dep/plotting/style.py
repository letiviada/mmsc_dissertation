import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def style_and_colormap(style='seaborn-v0_8', colormap='tab20', num_positions = None):
    """
    Set the plotting style and return the colormap.

    Parameters:
    style (str): The style to use for plotting.
    colormap (str): The name of the colormap to use.
    num_positions (int): Number of positions to generate colours for.

    Returns:
    colormap (matplotlib.colors.Colormap): The colormap to use for plotting.
    """
    plt.style.use(style)
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
    })
    color_map = plt.get_cmap(colormap,num_positions)
    colors = None
    if num_positions is not None:
        colors = color_map(np.linspace(0,1,num_positions))
    return color_map, colors

def create_legend_exactandnum(fig, values_dict,style = 'seaborn-v0_8', colormap='tab20'):
    """
    Create a colormap legend that includes in the same entry the color for the exact solution (black) and the numerical (color).

    Parameters:
    fig (matplotlib.figure.Figure): The figure to add the legend to.
    values_dict (dict): Dictionary of values for plotting.
    colormap (str): The name of the colormap to use.
    """
    num_positions = len(values_dict)
    _, colors = style_and_colormap(style = style, colormap = colormap, num_positions = num_positions)
    custom_lines = [(Line2D([0], [0], color='k', linewidth=2), Line2D([0], [0], color=color, linestyle='--', linewidth=2)) for color in colors]
    custom_labels = [f"{label}" for label in values_dict.values()]

    # Creating a combined legend
    fig.legend(handles=custom_lines, labels=custom_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='outside center right', borderaxespad=1, ncol=1)
def create_legend_times(fig, t_eval,style = 'seaborn-v0_8', colormap='tab20'):
    """
    Create a colormap legend that includes in the same entry the color for the exact solution (black) and the numerical (color).

    Parameters:
    fig (matplotlib.figure.Figure): The figure to add the legend to.
    values_dict (dict): Dictionary of values for plotting.
    colormap (str): The name of the colormap to use.
    """
    num_positions = len(t_eval)
    _, colors = style_and_colormap(style = style, colormap = colormap, num_positions = num_positions)
    custom_lines = [(Line2D([0], [0], color='k', linewidth=2), Line2D([0], [0], color=color, linestyle='--', linewidth=2)) for color in colors]
    custom_labels = [f"t={t}" for t in t_eval]

    # Creating a combined legend
    fig.legend(handles=custom_lines, labels=custom_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='outside center right', borderaxespad=1, ncol=1)