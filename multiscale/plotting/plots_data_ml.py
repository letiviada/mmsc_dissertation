import matplotlib.pyplot as plt
import seaborn as sns
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import numpy as np
import pandas as pd

def scatter_histogram(data,output):
    """
    Function that plots the scatter plot and histogram of the data for a given output.

    Parameters:
    ----------
    data (pd.DataFrame): the data to plot
    output (str): the output to plot

    Returns:
    --------
    fig (plt.Figure): the figure of the plot
    """
    _, colors = style_and_colormap(num_positions = 13, colormap = 'tab20b')#
    fig, ax = create_fig(1,2,dpi = 100)
    colors = colors.tolist()
    data_plot = data[['adhesivity', 'particle_size', output]] 
    sns.scatterplot(data_plot, x='adhesivity', y=output, ax=ax[0], size='particle_size', sizes=(150, 150),
                    hue='particle_size', palette=colors[:13], legend=False),

   # ax[0].legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncols=)

    sns.histplot(data = data, x = output, ax = ax[1], bins = 15, kde = True, color = colors[7])
    plt.tight_layout
    return fig

def view_data_all_outputs(data, outputs ):
    for output in outputs:
        output_name = output.replace(" ", "_")
        fig = scatter_histogram(data, output)
        save_figure(fig, f'regression/figures/data_/{output_name}_vs_alpha_beta')

def boxplot(data,outputs):
    """
    Function that plots the boxplot of the data for all the outputs.

    Parameters:
    ----------
    data (pd.DataFrame): the data to plot
    outputs (list): the list of outputs to plot

    Returns:
    --------
    fig (plt.Figure): the figure of the plot
    """
    _, colors = style_and_colormap(num_positions = 3, colormap = 'tab20b')
    fig, ax = create_fig(1,len(outputs),dpi = 100)
    colors = colors.tolist()
    for i,output in enumerate(outputs):
        data_plot = data[['Adhesivity', 'Particle Size', output]] 
        sns.boxplot(data = data_plot, y = output,ax =ax[i], color = colors[i])
        y_max = data[output].max() 
        ax[i].set_yticks(np.arange(0,y_max + 50, 50)) 
    plt.tight_layout()
    save_figure(fig, 'regression/figures/data_/box_all_outputs')
    return fig

def scatter_solutions(inputs,outputs,name,type_model):
    """
    Function that plots the solutions of the model and the actual values of the data

    Parameters:
    ----------
    inputs (pd.DataFrame): the inputs of the model
    outputs (pd.DataFrame): the outputs of the model
    name (str): the name of the output
    type_model (str): the type of model

    Returns:
    --------
    None
    """
    ps_unique_keys = inputs['particle_size'].unique()
    num_unique_keys = len(ps_unique_keys)
    sns.set_theme()
    fig, ax = create_fig(nrows = 1, ncols = 2 ,dpi = 100)
    _, colors = style_and_colormap(num_positions = num_unique_keys, colormap = 'tab20b')
    colors = colors.tolist()
    # Plot the results of the model
    ax[0].plot(outputs[name] , outputs[name] , color = colors[1])
    sns.scatterplot(x = outputs[name] , y = outputs['Prediction'] , ax = ax[0], color = colors[0])

    combined = pd.concat([
                inputs.assign(Solution = outputs[name], Type = 'Actual'),
                inputs.assign(Solution = outputs['Prediction'], Type = 'Prediction')
                
    ] )
    sns.scatterplot(data = combined,x = 'adhesivity', y = 'Solution', 
                    hue = 'particle_size', style = 'Type', 
                    palette = colors, ax = ax[1])
    ax[1].legend(bbox_to_anchor= (1.5,1.0), loc = 'upper right', ncols = 2) 
    plt.tight_layout()
    name_save = name.replace(" ", "_").lower()
    save_figure(fig, f'regression/figures/data_{type_model}/solution_{name_save}')
    #plt.show()
    return fig, ax

def opt_ml(full_data:pd.DataFrame, name:str, lines:bool, actual: bool, predictions: bool):
    # Prepare figure, style, and colours
    # ----------------------------------

    unique_keys = full_data['particle_size'].unique()
    num_unique_keys = len(unique_keys)
    _, colors = style_and_colormap(num_positions = num_unique_keys, colormap = 'tab20b')
    fig, ax = create_fig(nrows = 1, ncols = 1 ,dpi = 100)
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(unique_keys, colors)}

    # Plot the results of the model
    # -----------------------------
    name_pred = name.split('_time')[0] + '_predictions'
    if actual == True:
        sns.scatterplot(data = full_data, x = 'adhesivity', y = name, hue = 'particle_size', palette = color_mapping, ax = ax[0])
    if predictions  == True:
        for i, beta_value in enumerate(unique_keys):
            sorted_data = full_data[full_data['particle_size'] == beta_value].sort_values('adhesivity')
            if lines == True:
                ax[0].plot(sorted_data['adhesivity'], sorted_data[name_pred], marker='x', color=colors[i])
            else:
                ax[0].scatter(sorted_data['adhesivity'], sorted_data[name_pred], color=colors[i], marker = 'x')
    plt.show()
