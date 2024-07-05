import matplotlib.pyplot as plt
import seaborn as sns
from .create import create_fig
from .style import style_and_colormap
from .save import save_figure
import numpy as np
import pandas as pd

def scatter_histogram(data,output):
    _, colors = style_and_colormap(num_positions = 9, colormap = 'tab20b')#
    fig, ax = create_fig(1,2,dpi = 100)
    colors = colors.tolist()
    data_plot = data[['Adhesivity', 'Particle Size', output]] 
    sns.scatterplot(data_plot, x='Adhesivity', y=output, ax=ax[0], size='Particle Size', sizes=(150, 150),
                    hue='Particle Size', palette=colors[:7], legend='full'),

    ax[0].legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncols=4)

    sns.histplot(data = data, x = output, ax = ax[1], binwidth = 100, kde = True, color = colors[7])
    plt.tight_layout
    return fig

def view_data_all_outputs(data, outputs ):
    for output in outputs:
        output_name = output.replace(" ", "_")
        fig = scatter_histogram(data, output)
        save_figure(fig, f'regression/figures/data_large/{output_name}_vs_alpha_beta')

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
    save_figure(fig, 'regression/figures/data_large/box_all_outputs')
    return fig
    