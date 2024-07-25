import argparse
from utils import load_k_j, load_any
from plotting import plot_one_dim, save_figure

def main(alphas: list, betas: list,phi:float,name:str):
    # Initialize lists
    # ----------------
    k_values_list = []
    j_values_list = []
    u_values_list =[] 
    tau_values_list = []
    c_values_list = []

    # Get k and j values for each alpha and beta constant
    # ---------------------------------------------------
    if isinstance(betas, float):
        betas = [betas]
    if isinstance(alphas, float):
        alphas = [alphas]
    for beta in betas:    
        for alpha in alphas:
            # Load k_values, j_values, tau_eval
            k_values, j_values, tau_eval = load_k_j(alpha, beta,run = None,filename = 'multiscale/results/mono-dispersed/microscale/micro_results.json')
            darcy_velocity = load_any(alpha,beta, 'darcy_velocity', run = None, filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
            darcy_velocity = [0 if v < 0.1 else v for v in darcy_velocity] 
            concentration  = load_any(alpha,beta,'concentration',run=None,filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
            # Append to respective lists
            k_values_list.append(k_values)
            j_values_list.append(j_values)


            u_values_list.append(darcy_velocity)
            c_values_list.append(concentration[-1,:])
    tau_values_list.append(tau_eval)
    t_eval = load_any(alpha,beta,'time_eval', run = None, filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
    x_eval = load_any(alpha,beta,'x_eval',run = None,filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')

# Plot the results
    # ----------------
    fig_k = plot_one_dim(tau_values_list[0],k_values_list)
    fig_k.grid(False)
    save_figure(fig_k, f'multiscale/figures/mono-dispersed/microscale/permeability/k_{name}')
    fig_j = plot_one_dim(tau_values_list[0],j_values_list)
    fig_j.grid(False)
    save_figure(fig_j, f'multiscale/figures/mono-dispersed/microscale/adhesivity/j_{name}')
    fig_u = plot_one_dim(t_eval, [v for v in u_values_list if v != 0], funct_type='velocity')
    fig_u.grid(False)
    save_figure(fig_u,f'multiscale/figures/mono-dispersed/macroscale/darcy_velocity/varying_parameters/darcy_velocity_{name}')
    fig_c = plot_one_dim(x_eval,c_values_list)
    fig_c.grid(False)
    save_figure(fig_c,f'multiscale/figures/mono-dispersed/macroscale/concentration/varying_parameters/concentration_{name}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', nargs = '+',type=float, help='List of alphas')
    parser.add_argument('--betas',nargs = '+', type=float, help='List of betas')
    parser.add_argument('--phis',nargs = '+', type=float, help='List of betas')
    parser.add_argument('--name', type=str, help='Name of the file')
    args = parser.parse_args()

    main(args.alphas, args.betas, args.phis, args.name)
