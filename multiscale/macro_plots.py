import argparse
import numpy as np
from plotting import plot_time, save_figure, plot_one_dim
import matplotlib.pyplot as plt
from utils import load_any
from tqdm import tqdm


def main(alpha,beta,phi,filename,parameter):
    # Load values
    t_eval = load_any(alpha,beta,'time_eval',filename)
    x_eval = load_any(alpha,beta,'x_eval',filename)
     # Time points of interest
    time_points = [0,10, 50, 100, 250, 500, 750, 1000, 1250,1500]
    # Find indices of the time points in t_eval
    indices = [np.abs(t_eval - t_point).argmin() for t_point in time_points]

    param = load_any(alpha,beta,parameter,filename)
    if parameter == 'darcy_velocity':
        u=param
        fig =  plot_one_dim(t_eval, [u], title='Darcy Velocity')
        save_figure(fig, f'multiscale/figures/macroscale/darcy_velocity/alpha_{alpha}/darcy_velocity_alpha_{alpha}_beta_{beta}_phi_{phi}')
        plt.close(fig)
    elif parameter == 'reactivity':
        filtered_t_eval = t_eval[indices]
        filtered_psi = param[indices, :]
        fig = plot_time(filtered_t_eval, x_eval,filtered_psi, title='Reactivity')
        save_figure(fig, f'multiscale/figures/macroscale/reactivity/reactivity_time/alpha_{alpha}/reactivity_alpha_{alpha}_beta_{beta}_phi_{phi}')
        fig2 = plot_one_dim(t_eval, [param[:,1]], title='Reactivity')
        save_figure(fig2, f'multiscale/figures/macroscale/reactivity/reactivity/alpha_{alpha}/reactivity_alpha_{alpha}_beta_{beta}_phi_{phi}')
        plt.close(fig)
        plt.close(fig2)
    else:
        filtered_t_eval = t_eval[indices]
        filtered_param = param[indices, :]

        fig = plot_time(filtered_t_eval, x_eval, filtered_param, title=parameter)
        save_figure(fig, f'multiscale/figures/macroscale/{parameter}/alpha_{alpha}/{parameter}_alpha_{alpha}_beta_{beta}_phi_{phi}') 
        plt.close(fig)


# Function to generate alpha beta pairs
def alpha_beta_pairs(alpha_values, beta_values):
    alpha_beta_pairs = [(alpha, beta) for alpha in alpha_values for beta in beta_values]
    return alpha_beta_pairs



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot macro results')
    parser.add_argument('--alpha_values', nargs='+', type=float, help='List of alpha values')
    parser.add_argument('--beta_values', nargs='+', type=float, help='List of beta values')
    parser.add_argument('--phi', type=float, help='Value of phi')
   # parser.add_argument('--parameter', nargs = '+',type=str, help='Parameter to plot')
    args = parser.parse_args()

    alpha_beta_pairs = alpha_beta_pairs(args.alpha_values, args.beta_values)
    parameters=["auxiliar_variable", "concentration", "darcy_velocity",
                 "permeability", "adhesivity", "reactivity"]
    parameters=["darcy_velocity","reactivity"]
    for alpha, beta in tqdm(alpha_beta_pairs):
        for parameter in tqdm(parameters):
            main(alpha, beta, args.phi,f'multiscale/results/macroscale/macro_results_phi_{args.phi}.json',parameter)
