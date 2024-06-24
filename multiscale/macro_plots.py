   # Plot solutions
    # -----------------
    #fig = plot_time(model.t_eval, model.x_eval, model.c, title='Concentration')
    #fig2 = plot_time(model.t_eval, model.x_eval, model.tau, title='Auxiliar variable')
    #fig3 = plot_time(model.t_eval, model.x_eval, model.k, title='Permeability')
    #fig4 = plot_time(model.t_eval, model.x_eval, model.j, title='Adhesivity')
    #fig5 = plot_one_dim(model.t_eval, [model.u], title='Darcy Velocity')
    #fig6 = plot_time(model.t_eval, model.x_eval,model.psi, title='Reactivity')
    #fig7 = plot_one_dim(model.t_eval, [model.psi[:,1]], title='Reactivity')

    # Save figures
    # ---------------
    #save_figure(fig, 'multiscale/figures/concentration/c')
    #save_figure(fig2, 'multiscale/figures/tau/tau')
    #save_figure(fig3, 'multiscale/figures/permeability/k')
    #save_figure(fig4, 'multiscale/figures/adhesivity/j')
    #save_figure(fig5, 'multiscale/figures/velocity/u')
    #save_figure(fig6, 'multiscale/figures/reactivity/psi')
    #save_figure(fig7, 'multiscale/figures/reactivity/psi_time')

from plotting import plot_time, save_figure, plot_one_dim
from utils import load_any
def main(alpha,beta):
    # Load values
    t_eval = load_any(alpha,beta,'time_eval')
    x_eval = load_any(alpha,beta,'x_eval')
    c = load_any(alpha,beta,'concentration')
    tau = load_any(alpha,beta,'auxiliar_variable')
    k = load_any(alpha,beta,'permeability')
    j = load_any(alpha,beta,'adhesivity')
    u = load_any(alpha,beta,'darcy_velocity')
    psi = load_any(alpha,beta,'reactivity')

    # Plot solutions
    # -----------------
    fig = plot_time(t_eval, x_eval, c, title='Concentration')
    fig2 = plot_time(t_eval, x_eval, tau, title='Auxiliar variable')
    fig3 = plot_time(t_eval, x_eval, k, title='Permeability')
    fig4 = plot_time(t_eval, x_eval, j, title='Adhesivity')
    fig5 = plot_one_dim(t_eval, [u], title='Darcy Velocity')
    fig6 = plot_time(t_eval, x_eval,psi, title='Reactivity')
    fig7 = plot_one_dim(t_eval, [psi[:,1]], title='Reactivity')

    # Save figures
    # ---------------
    save_figure(fig, f'multiscale/figures/macroscale/concentration/c_{alpha}_{beta}')
    save_figure(fig2, f'multiscale/figures/macroscale/tau/tau_{alpha}_{beta}')
    save_figure(fig3, f'multiscale/figures/macroscale/permeability/k_{alpha}_{beta}')
    save_figure(fig4, f'multiscale/figures/macroscale/adhesivity/j_{alpha}_{beta}')
    save_figure(fig5, f'multiscale/figures/macroscale/velocity/u_{alpha}_{beta}')
    save_figure(fig6, f'multiscale/figures/macroscale/reactivity/psi_{alpha}_{beta}')
    save_figure(fig7, f'multiscale/figures/macroscale/reactivity/psi_time_{alpha}_{beta}')

if __name__ == '__main__':
    alpha, beta = 1.0, 0.01
    main(alpha,beta)