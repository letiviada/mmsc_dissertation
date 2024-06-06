import matplotlib.pyplot as plt

def create_fig(nrows, ncols, title = None, figsize=(15, 8), dpi=300):
    """
    Create a figure and axes.

    Parameters:
    nrows (int): Number of rows of subplots.
    ncols (int): Number of columns of subplots.
    title (str): Title of the figure created.
    figsize (tuple): Size of the figure.
    dpi (int): Dots per inch of the figure.

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    axes (np.ndarray): Array of Axes objects.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    if title is not None:
        fig.suptitle(title)
    return fig, axes.flatten()