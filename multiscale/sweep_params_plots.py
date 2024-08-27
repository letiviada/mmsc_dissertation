import argparse
from utils import load_k_j, load_any
from plotting import plot_one_dim, save_figure
import numpy as np
def main(alphas: list, betas: list,phi:float,name:str):
    # Initialize lists
    # ----------------
    k_values_list = []
    j_values_list = []
    u_values_list =[] 
    tau_values_list = []
    c_values_list = []
    r_t_values_list = []
    phi = 2.0
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
            prop_values = np.divide(k_values, j_values)
            #darcy_velocity = load_any(alpha,beta, 'darcy_velocity', run = None, filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
            #concentration  = load_any(alpha,beta,'concentration',run=None,filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
            reactivity = load_any(alpha,beta,'reactivity',run=None,filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
            # Append to respective lists
            k_values_list.append(prop_values)
           # j_values_list.append(j_values)
           # u_values_list.append(darcy_velocity)
           # c_values_list.append(concentration[-1,:])
            r_t_values_list.append(reactivity[:,-1])
    tau_values_list.append(tau_eval)
    t_eval = load_any(alpha,beta,'time_eval', run = None, filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')
    x_eval = load_any(alpha,beta,'x_eval',run = None,filename = f'multiscale/results/mono-dispersed/macroscale/macro_results_phi_{phi}.json')

# Plot the results
    # ----------------
    fig_k, ax_k = plot_one_dim(tau_values_list[0],k_values_list)
    ax_k[0].set_yticks([0,0.2, 0.4, 0.6, 0.8,1.0])
    ax_k[0].set_xlabel('$s$')
    ax_k[0].set_ylabel(r'$j/k$')
    save_figure(fig_k, f'multiscale/figures/mono-dispersed/microscale/prop/k_j_{name}')
   # fig_j, ax_j = plot_one_dim(tau_values_list[0],j_values_list)
   # ax_j[0].set_xlabel('$s$')
   # ax_j[0].set_ylabel('$j$')
   # save_figure(fig_j, f'multiscale/figures/mono-dispersed/microscale/adhesivity/j_{name}')
   # fig_u, ax_u = plot_one_dim(t_eval, u_values_list, funct_type='velocity')
   ## ax_u[0].set_xlabel('$t$')
   # ax_u[0].set_ylabel('$u$')
    #save_figure(fig_u,f'multiscale/figures/mono-dispersed/macroscale/darcy_velocity/varying_parameters/darcy_velocity_{name}')
   # fig_c,ax_c = plot_one_dim(x_eval,c_values_list)
  #  ax_c[0].set_xticks([0,0.2, 0.4, 0.6, 0.8,1.0])
   # ax_c[0].set_xlabel('$x$')
  #  ax_c[0].set_ylabel('$c$')
    #save_figure(fig_c,f'multiscale/figures/mono-dispersed/macroscale/concentration/varying_parameters/concentration_{name}')
    #fig_r_t, ax_r_t = plot_one_dim(t_eval,r_t_values_list)
    #ax_r_t[0].set_xlabel('$t$')
    #ax_r_t[0].set_ylabel('$\psi$')
    #ax_r_t[0].set_yticks([0,0.2, 0.4, 0.6, 0.8,1.0,1.2])
    #save_figure(fig_r_t,f'multiscale/figures/mono-dispersed/macroscale/reactivity/varying_parameters/reactivity_{name}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', nargs = '+',type=float, help='List of alphas')
    parser.add_argument('--betas',nargs = '+', type=float, help='List of betas')
    parser.add_argument('--phis',nargs = '+', type=float, help='List of betas')
    parser.add_argument('--name', type=str, help='Name of the file')
    args = parser.parse_args()

    main(args.alphas, args.betas, args.phis, args.name)
