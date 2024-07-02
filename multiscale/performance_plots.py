import argparse
import numpy as np
from plotting import plot_time, save_figure, plot_one_dim
import matplotlib.pyplot as plt
from utils import load_any
from tqdm import tqdm


def main(alpha,beta,phi,filename,parameter):
    t_eval = load_any(alpha,beta,'time',run = 0,filename=filename)
    param = load_any(alpha,beta,parameter,run = 0,filename=filename)
    fig =  plot_one_dim(t_eval,[param])
    save_figure(fig, f'multiscale/figures/mono-dispersed/performance_indicators/{parameter}/{parameter}_alpha_{alpha}_beta_{beta}_phi_{phi}')
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
    parameters=["efficiency","throughput"]
    for alpha, beta in tqdm(alpha_beta_pairs):
        for parameter in tqdm(parameters):
            main(alpha, beta, args.phi,f'multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_{args.phi}.json',parameter)
