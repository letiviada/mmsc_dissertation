import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.utils import save_results, load_any
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
def plot(pressure,dpdx, x_eval):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(pressure.shape[0]):
        ax1.plot(x_eval, pressure[i,:])
        ax2.plot(x_eval, dpdx[i,:])
    ax1.set_xlabel('x')
    ax1.set_ylabel('Pressure')

    plt.show()
    return fig


p,dpdx,x_eval = pressure(1.0, 0.01, 4.0, None, 'multiscale/results/mono-dispersed/macroscale/macro_results_phi_4.0.json')
fig = plot(p,dpdx,x_eval)
