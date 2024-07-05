import matplotlib.pyplot as plt
import seaborn as sns
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import numpy as np
import pandas as pd

def scatter_histogram(data,output):
    _, colors = style_and_colormap(num_positions = 13, colormap = 'tab20b')#
    fig, ax = create_fig(1,2,dpi = 100)
    colors = colors.tolist()
    data_plot = data[['Adhesivity', 'Particle Size', output]] 
    sns.scatterplot(data_plot, x='Adhesivity', y=output, ax=ax[0], size='Particle Size', sizes=(150, 150),
                    hue='Particle Size', palette=colors[:13], legend=False),

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

def scatter_solutions(inputs,outputs,name):
    sns.set_theme()
    fig, ax = create_fig(nrows = 1, ncols = 2 ,dpi = 100)
    _, colors = style_and_colormap(num_positions = 13, colormap = 'tab20b')
    colors = colors.tolist()
    # Plot the results of the model
    ax[0].plot(outputs['Termination time'] , outputs['Termination time'] , color = colors[1])
    sns.scatterplot(x = outputs['Termination time'] , y = outputs['Prediction'] , ax = ax[0], color = colors[0])

    combined = pd.concat([
                inputs.assign(Termination_time = outputs['Termination time'], Type = 'Actual'),
                inputs.assign(Termination_time = outputs['Prediction'], Type = 'Prediction')
                
    ] )
    sns.scatterplot(data = combined,x = 'Adhesivity', y = 'Termination_time', 
                    hue = 'Particle Size', style = 'Type', 
                    palette = colors, ax = ax[1])
    ax[1].legend(bbox_to_anchor= (1.05,1.0), loc = 'upper right', ncols = 2) 
    plt.tight_layout()
    save_figure(fig, f'regression/figures/data_/solution_regression_{name}')
    plt.show()