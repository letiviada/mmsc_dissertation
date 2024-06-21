import json
import argparse
import numpy as np
from utils import load_k_j
from plotting import plot_one_dim, save_figure

def main(alphas: list, betas: float):
    # Initialize lists
    # ----------------
    alpha_values = [] 
    k_values_list = []
    j_values_list = []
    tau_values_list = []

    # Get k and j values for each alpha and beta constant
    # ---------------------------------------------------
    for alpha in alphas:
       #for beta in betas:
            alpha_values.append(alpha)

            #beta_values.append(beta)

            # Load k_values, j_values, tau_eval
            k_values, j_values, tau_eval = load_k_j(betas, alpha)

            # Append to respective lists
            k_values_list.append(k_values)
            j_values_list.append(j_values)
    tau_values_list.append(tau_eval)
    # Plot the results
    # ----------------
    fig_k = plot_one_dim(tau_values_list[0],k_values_list, title = 'Permeability', x_eval_type='tau')
    save_figure(fig_k, 'multiscale/figures/microscale/permeability/k2')
    fig_j = plot_one_dim(tau_values_list[0],j_values_list, title = 'Adhesivity', x_eval_type='tau')
    save_figure(fig_j, 'multiscale/figures/microscale/adhesivity/j2')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', nargs = '+',type=float, help='List of alphas')
    parser.add_argument('--betas', type=float, help='List of betas')
    args = parser.parse_args()

    main(args.alphas, args.betas)