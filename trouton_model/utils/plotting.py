import numpy as np
import matplotlib.pyplot as plt

def plot_results(x_eval, t_eval, x_array, option='time', point=None, variable='thickness', ax=None, fig=None):
    """
    Plots the results against time or space, optionally on given subplot axes.

    Args:
        x_eval (np.ndarray): The evaluation points for time or space.
        t_eval (np.ndarray): The time evaluation points.
        x_array (np.ndarray): The results array with shape (nt, nx).
        option (str): 'time' or 'space' to plot time points or space points.
        point (float): The specific point to plot. If None, plots multiple points.
        variable (str): The variable being plotted ('thickness' or 'axial velocity').
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on. If None, creates new axes.
        fig (matplotlib.figure.Figure, optional): Figure to plot on. If None, creates a new figure.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    nt = x_array[:,0].shape[0]
    nx = x_array[0,:].shape[0]
    if option == 'time':
        if point is not None:
            # Find the index closest to the specified time point
            time_index = (np.abs(x_eval - point)).argmin()
            ax.plot(x_eval, x_array[time_index,:], label=f't = {t_eval[time_index]:.2f}')
        else:
            step_size = max(1, nt // 10)  # Ensure we have a reasonable step size
            for i in range(0, nt, step_size):
                ax.plot(x_eval, x_array[i, :], label=f't = {t_eval[i]:.1f}')
        title = f'{variable.capitalize()} at different time points'
            
    elif option == 'space':
        if point is not None:
            # Find the index closest to the specified time point
            space_index = (np.abs(x_eval - point)).argmin()
            ax.plot(t_eval, x_array[space_index,:], label=f'x = {x_eval[space_index]:.1f}')
        else:
            step_size = max(1, nx // 5)  # Ensure we have a reasonable step size
            for i in range(0, nx, step_size):
                ax.plot(t_eval, x_array[:, i], label=f'x = {x_eval[i]:.2f}')
        title = f'Evolution of {variable} at specific spatial points'
        
    ax.set_title(title)
    ax.set_xlabel('Space' if option == 'time' else 'Time')
    ax.set_ylabel(variable.capitalize())
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fancybox=True, shadow=True)
    fig.tight_layout()
    return fig, ax