import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.utils import save_results, load_any
from multiscale.plotting import pressure_plot
from get_pressure import Pressure
import matplotlib.pyplot as plt


def pressure(alpha,beta,phi,run,filename):
    t_eval = load_any(alpha,beta,'time_eval',run,filename)
    x_eval = load_any(alpha,beta,'x_eval',run,filename)
    k = load_any(alpha,beta,'permeability',run,filename)
    u = load_any(alpha,beta,'darcy_velocity',run,filename)
    pressure = Pressure(t_eval,x_eval,k,u)
    p = pressure.pressure()
    dpdx = pressure.pressure_grad()
    return p, dpdx, x_eval   


p,dpdx,x_eval = pressure(alpha = 0.6, beta = 0.01,
                          phi = 4.0, run = None, filename = 'multiscale/results/mono-dispersed/macroscale/macro_results_phi_4.0.json')
fig_p, fig_dpdx = pressure_plot(p,dpdx,x_eval)