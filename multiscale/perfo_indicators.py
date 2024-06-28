from utils import FilterPerformance
import numpy as np
from utils import load_any

def main(alpha,beta,phi,filename):
    # Load values
    t_eval = load_any(alpha,beta,'time_eval',filename)
    velocity = load_any(alpha,beta,'darcy_velocity',filename)
    concentration = load_any(alpha,beta,'concentration',filename)
    filter_performance = FilterPerformance(t_eval=t_eval, u=velocity,c=concentration)

    termination_time = filter_performance.termination_time(mu=0.1)
    throughput_time = np.linspace(0,termination_time,3)
    throuput = filter_performance.throughput(tf=throughput_time)
    efficiency = filter_performance.efficiency()
    print(f'Termination time: {termination_time}')
    print(f'Throughput: {throuput}')
    print(f'Efficiency: {efficiency}')

if __name__ == '__main__':
    alpha, beta = 0.1, 0.1
    phi = 0.1
    main(alpha, beta, phi,f'multiscale/results/macroscale/macro_results_phi_{phi}.json')
    




