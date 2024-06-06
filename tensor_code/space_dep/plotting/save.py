def save_figure(fig, path):
    """
    Saves the figure to the specified path in both SVG and PDF formats.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        path (str): The directory path where the figure should be saved.
    """
    fig.savefig(f"{path}.svg")
    fig.savefig(f"{path}.pdf")