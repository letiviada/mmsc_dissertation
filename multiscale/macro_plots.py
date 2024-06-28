from plotting import plot_time, save_figure, plot_one_dim
import matplotlib.pyplot as plt
from utils import load_any
from tqdm import tqdm
def main(alpha,beta,phi,filename):
    # Load values
    t_eval = load_any(alpha,beta,'time_eval',filename)
    x_eval = load_any(alpha,beta,'x_eval',filename)
    c = load_any(alpha,beta,'concentration',filename)
    tau = load_any(alpha,beta,'auxiliar_variable',filename)
    k = load_any(alpha,beta,'permeability',filename)
    j = load_any(alpha,beta,'adhesivity',filename)
    u = load_any(alpha,beta,'darcy_velocity',filename)
    psi = load_any(alpha,beta,'reactivity',filename)

    # Plot solutions
    # -----------------
    fig = plot_time(t_eval[100:150], x_eval, c[100:150], title='Concentration')
    fig2 = plot_time(t_eval[100:150], x_eval, tau[100:150], title='Auxiliar variable')
    #fig3 = plot_time(t_eval, x_eval, k, title='Permeability')
    #fig4 = plot_time(t_eval, x_eval, j, title='Adhesivity')
    fig5 = plot_one_dim(t_eval, [u], title='Darcy Velocity')
    #fig6 = plot_time(t_eval, x_eval,psi, title='Reactivity')
    #fig7 = plot_one_dim(t_eval, [psi[:,1]], title='Reactivity')

    # Save figures
    # ---------------
    save_figure(fig, f'multiscale/figures/macroscale/concentration/c_alpha_{alpha}_beta_{beta}_phi_{phi}')
    save_figure(fig2, f'multiscale/figures/macroscale/tau/alpha_{alpha}_beta_{beta}/tau_alpha_{alpha}_beta_{beta}_phi_{phi}')
    #save_figure(fig3, f'multiscale/figures/macroscale/permeability/k_{alpha}_{beta}')
    #save_figure(fig4, f'multiscale/figures/macroscale/adhesivity/j_{alpha}_{beta}')
    save_figure(fig5, f'multiscale/figures/macroscale/velocity/alpha_{alpha}_beta_{beta}/u_alpha_{alpha}_beta_{beta}_phi_{phi}')
    #save_figure(fig6, f'multiscale/figures/macroscale/reactivity/psi_{alpha}_{beta}')
    #save_figure(fig7, f'multiscale/figures/macroscale/reactivity/psi_time_{alpha}_{beta}')
    plt.close(fig2)
    plt.close(fig5)
if __name__ == '__main__':
    #alpha_values = [0.3]  # List of alpha values
    #beta_values = [0.03]  # List of beta values
    #alpha_beta_pairs = [(alpha, beta) for alpha in alpha_values for beta in beta_values]
    alpha_beta_pairs = [(0.3,0.03),(0.5,0.05),(0.7,0.07)]
    #phi_values = [0.9,0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  #
    #phi_values = [0.9]
    for alpha, beta in tqdm(alpha_beta_pairs):
        #for phi in tqdm(phi_values):
            #main(alpha, beta, phi, filename='multiscale/results/macroscale/macro_results_phi_{:.1f}.json'.format(phi))
            main(alpha, beta, 0.7, filename='multiscale/results/macroscale/macro_results_phi_0.7.json')
   # phi_values = [0.7, 0.6, 0.5, 0.8]  # List of phi values


