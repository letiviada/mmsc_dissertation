import argparse
from utils import load_k_j
from plotting import plot_one_dim, save_figure

def main(alphas: list, betas: list,name:str):
    # Initialize lists
    # ----------------
    k_values_list = []
    j_values_list = []
    tau_values_list = []

    # Get k and j values for each alpha and beta constant
    # ---------------------------------------------------
    if isinstance(betas, float):
        betas = [betas]
    if isinstance(alphas, float):
        alphas = [alphas]
    for beta in betas:    
        for alpha in alphas:
            # Load k_values, j_values, tau_eval
            k_values, j_values, tau_eval = load_k_j(alpha, beta,0)

            # Append to respective lists
            k_values_list.append(k_values)
            j_values_list.append(j_values)
    tau_values_list.append(tau_eval)
    # Plot the results
    # ----------------
    fig_k = plot_one_dim(tau_values_list[0],k_values_list, title = 'Permeability', x_eval_type='tau')
    save_figure(fig_k, f'multiscale/figures/microscale/permeability/k_{name}')
    fig_j = plot_one_dim(tau_values_list[0],j_values_list, title = 'Adhesivity', x_eval_type='tau')
    save_figure(fig_j, f'multiscale/figures/microscale/adhesivity/j_{name}')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', nargs = '+',type=float, help='List of alphas')
    parser.add_argument('--betas',nargs = '+', type=float, help='List of betas')
    parser.add_argument('--name', type=str, help='Name of the file')
    args = parser.parse_args()

    main(args.alphas, args.betas, args.name)