from plotting import plot_time, save_figure, plot_one_dim
from utils import load_any
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
    #fig = plot_time(t_eval, x_eval, c, title='Concentration')
    fig2 = plot_time(t_eval[-2:], x_eval, tau[-2:], title='Auxiliar variable')
    #fig3 = plot_time(t_eval, x_eval, k, title='Permeability')
    #fig4 = plot_time(t_eval, x_eval, j, title='Adhesivity')
    fig5 = plot_one_dim(t_eval, [u], title='Darcy Velocity')
    #fig6 = plot_time(t_eval, x_eval,psi, title='Reactivity')
    #fig7 = plot_one_dim(t_eval, [psi[:,1]], title='Reactivity')

    # Save figures
    # ---------------
    #save_figure(fig, f'multiscale/figures/macroscale/concentration/c_{alpha}_{beta}')
    save_figure(fig2, f'multiscale/figures/macroscale/tau/tau_alpha_{alpha}_beta_{beta}_phi_{phi}')
    #save_figure(fig3, f'multiscale/figures/macroscale/permeability/k_{alpha}_{beta}')
    #save_figure(fig4, f'multiscale/figures/macroscale/adhesivity/j_{alpha}_{beta}')
    save_figure(fig5, f'multiscale/figures/macroscale/velocity/u_alpha_{alpha}_beta_{beta}_phi_{phi}')
    #save_figure(fig6, f'multiscale/figures/macroscale/reactivity/psi_{alpha}_{beta}')
    #save_figure(fig7, f'multiscale/figures/macroscale/reactivity/psi_time_{alpha}_{beta}')

if __name__ == '__main__':
    alpha, beta = 0.3,0.04
    phi=0.7
    main(alpha,beta, phi,filename='multiscale/results/macroscale/macro_results_phi_0.7.json')
    phi=0.6
    main(alpha,beta, phi,filename='multiscale/results/macroscale/macro_results_phi_0.6.json')
    phi=0.5
    main(alpha,beta,phi,filename='multiscale/results/macroscale/macro_results_phi_0.5.json')
    phi=0.8
    main(alpha,beta, phi,filename='multiscale/results/macroscale/macro_results_phi_0.8.json')
