import os

def save_figure(fig, path):
    """
    Saves the figure to the specified path in both SVG and PDF formats.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        path (str): The directory path where the figure should be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the figure
    fig.savefig(f"{path}.svg")
    fig.savefig(f"{path}.pdf")
