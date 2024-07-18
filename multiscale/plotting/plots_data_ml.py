import matplotlib.pyplot as plt
import seaborn as sns
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import numpy as np
import pandas as pd
from scipy.stats import linregress
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

def scatter_solutions(inputs,outputs,name,type_model,name_eval):
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
    _, colors = style_and_colormap(num_positions = num_unique_keys, colormap = 'tab20b')
    fig, ax = create_fig(nrows = 1, ncols = 2 ,dpi = 100)
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(ps_unique_keys, colors)}
    # Plot the results of the model
    ax[0].plot(outputs[name] , outputs[name] , color = colors[1])
    sns.scatterplot(x = outputs[name] , y = outputs['prediction'] , ax = ax[0], color = colors[0])

    combined = pd.concat([
                inputs.assign(Solution = outputs[name], Type = 'actual'),
                inputs.assign(Solution = outputs['prediction'], Type = 'prediction')
                
    ] )
    sns.scatterplot(data = combined,x = 'adhesivity', y = 'Solution', 
                    hue = 'particle_size', style = 'Type', 
                    palette = color_mapping, ax = ax[1])
    ax[1].legend(bbox_to_anchor= (1.5,1.0), loc = 'upper right', ncols = 2) 
    plt.tight_layout()
    name_save = name.replace(" ", "_").lower()
    save_figure(fig, f'regression/figures/data_{type_model}/{name_save}/solution_{name_save}_{name_eval}')
    return fig, ax

def model_plot_with_lines_and_scatter(inputs, outputs, name, type_model, data_lines,data_model):
    ps_unique_keys = inputs['particle_size'].unique()
    if data_lines is not None:
        ps2_unique_keys = data_lines['particle_size'].unique()
        ps_unique_keys = np.unique(ps_unique_keys)
        ps2_unique_keys = np.unique(ps2_unique_keys)
        unique_keys = np.unique(np.concatenate((ps_unique_keys, ps2_unique_keys)))
    else:
        unique_keys = ps_unique_keys
    num_unique_keys = len(unique_keys)
    _, colors = style_and_colormap(num_positions = num_unique_keys, colormap = 'tab20b')
    fig, ax = create_fig(nrows = 1, ncols = 1 ,dpi = 100)
    
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(unique_keys, colors)}
    # Plot the results of the model

    combined = pd.concat([
                inputs.assign(Solution = outputs[name])
    ] )
    combined = combined.sort_values('particle_size')

    combined2 = pd.concat([
                inputs.assign(Solution = outputs['prediction'])
    ] )
    data_model2 = data_model[data_model['particle_size'].isin(ps2_unique_keys)]
    combined2 = combined2[combined2['particle_size'].isin(ps2_unique_keys)]
    combined = combined[combined['particle_size'].isin(ps2_unique_keys)]
    sns.scatterplot(data = data_model2, x = 'adhesivity', y = name,hue = 'particle_size', palette = color_mapping, ax = ax[0])
    sns.scatterplot(data = combined2, x = 'adhesivity', y = 'Solution', marker = 'x', 
                   color = 'k', ax = ax[0], linewidths=2)
    sns.scatterplot(data = combined,x = 'adhesivity', y = 'Solution', 
                   color = 'red', ax = ax[0])
    for i, beta_value in enumerate(ps2_unique_keys):
        sorted_data_line = data_lines[data_lines['particle_size'] == beta_value].sort_values('adhesivity')
        ax[0].plot(sorted_data_line['adhesivity'], sorted_data_line[name], color=color_mapping[beta_value])
    
    plt.tight_layout()
    name_save = name.replace(" ", "_").lower()
    save_figure(fig, f'regression/figures/data_{type_model}/{name_save}/solution_{name_save}_with_lines')
    plt.show()


def opt_ml(full_data:pd.DataFrame, name:str, actual: bool, predictions: bool,lines:bool,particle_sizes : list = None,data_line:pd.DataFrame = None,type_data:str = 'standard'):
    # Prepare figure, style, and colours
    # ----------------------------------
    if particle_sizes is None:
        unique_keys = full_data['particle_size'].unique()
    else:
        unique_keys = particle_sizes
    if data_line is not None:
        unique_keys_pred = data_line['particle_size'].unique()
        num_unique_keys = max(len(unique_keys),len(unique_keys_pred))
    else: 
        num_unique_keys = len(unique_keys)
    _, colors = style_and_colormap(num_positions = num_unique_keys, colormap = 'tab20b')
    fig, ax = create_fig(nrows = 1, ncols = 1 ,dpi = 100)
    colors = colors.tolist()
    color_mapping = {key: color for key, color in zip(unique_keys, colors)}
    full_data = full_data.sort_values('particle_size')
    # Plot the results of the model
    # -----------------------------
    name_pred = name.split('_time')[0] + '_predictions'
    if actual == True:
        filtered_data = full_data[full_data['particle_size'].isin(unique_keys)]
        sns.scatterplot(data=filtered_data, x='adhesivity', y=name, hue='particle_size', palette=color_mapping, ax=ax[0])
    if predictions  == True:
        for i, beta_value in enumerate(unique_keys):
            sorted_data = full_data[full_data['particle_size'] == beta_value].sort_values('adhesivity')
            sorted_data_line = data_line[data_line['particle_size'] == beta_value].sort_values('adhesivity')
            if lines == True:
                ax[0].plot(sorted_data_line['adhesivity'], sorted_data_line[name_pred], color=colors[i])
            elif lines == True and data_line is None:
                raise ValueError('Please provide the data for the lines')
            else:
                ax[0].scatter(sorted_data['adhesivity'], sorted_data[name_pred], color=colors[i], marker='x')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[0].legend(title='Particle Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if predictions == True and lines == True:
        save_figure(fig, f'regression/figures/optimization/{type_data}/{name}/opt_{name}_with_predictions_and_lines')
    elif predictions == True and lines == False:
        save_figure(fig, f'regression/figures/optimization/{type_data}/{name}/opt_{name}_with_predictions')
    else:
        save_figure(fig, f'regression/figures/optimization/{type_data}/{name}/opt_{name}_actual_data')
    plt.show()

def make_loglog(data:pd.DataFrame,name:str,betas:list,type_data:str):
    """
    Make log-log plot to compare the data for the different particle sizes.

    Parameters:
    -----------
    data (pd.DataFrame): the data to plot
    name (str): the name of the output to plot
    betas (list): the list of particle sizes to plot

    Returns:
    --------
    None
    """
    num_unique_keys = len(betas)
    _, colors = style_and_colormap(num_positions = num_unique_keys, colormap = 'tab20b')
    fig, ax = create_fig(nrows = 1, ncols = 1 ,dpi = 100)
    colors = colors.tolist()
    for i,beta in enumerate(betas):
        data_ratio_ordered = data[data['particle_size'] == beta].sort_values('adhesivity')
        ax[0].loglog(data_ratio_ordered[data_ratio_ordered['particle_size'] == beta]['adhesivity'], 
                  data_ratio_ordered[data_ratio_ordered['particle_size'] == beta][name],
                  color = colors[i])
        log_stickiness = np.log10(data_ratio_ordered[data_ratio_ordered['particle_size'] == beta]['adhesivity'])
        log_y = np.log10(data_ratio_ordered[data_ratio_ordered['particle_size'] == beta][name])
        slope, intercept, r_value, p_value, std_err = linregress(log_stickiness, log_y)
        order = abs(slope)
        ax[0].loglog(data_ratio_ordered[data_ratio_ordered['particle_size'] == beta]['adhesivity'], 
                  10**(intercept + 0.005) * data_ratio_ordered[data_ratio_ordered['particle_size'] == beta]['adhesivity']**slope,
                  color = colors[i], linestyle = '--', label = f'Beta {beta},Order {order:.2f}')
    ax[0].set_xlabel('Stickiness')
    ax[0].set_ylabel('Volume Liquid Time 400')
    ax[0].legend(title='Particle Size', bbox_to_anchor=(1.05, 1), loc='upper left', ncols =2)
    plt.tight_layout()
    save_figure(fig, f'regression/figures/optimization/{type_data}/{name}/loglog_{name}')


   