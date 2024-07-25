import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation/')
from multiscale.plotting import plot_adhesivity
import matplotlib.pyplot as plt

throughput = 100
# load data
results_sum_csv = pd.read_csv(f'regression/opt_throughput/data/throughput_{throughput}/results_sum.csv')
results_product_csv = pd.read_csv(f'regression/opt_throughput/data/throughput_{throughput}/optimum_values_product.csv')
data_csv = pd.read_csv(f'regression/opt_throughput/data/throughput_{throughput}/data_for_sums.csv')

results_sum_df = pd.DataFrame(results_sum_csv)
results_product_df = pd.DataFrame(results_product_csv)
data_df = pd.DataFrame(data_csv)

results_product_df = results_product_df[results_product_df['particle_size'] == 0.06]

# Plot the data
fig, ax = plot_adhesivity(data_df, output = f'avg_retained_particles_throughput_{throughput}', particle_sizes = [0.06], save = False)
fig, ax = plot_adhesivity(data_df, output = f'time_throughput_{throughput}', particle_sizes = [0.06], save = False)
#plot_optimum(opt, particle_size,actual = True, predictions=False, save = False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.scatter(results_sum_df['n'], results_sum_df['adhesivity'], label = 'Sum')
ax1.set_xlabel('Importance of time')
ax1.set_ylabel('Optimal adhesivity')
ax1.set_title('Sum of time and retained particles')
ax2.scatter(results_product_df['n'], results_product_df['adhesivity'], label = 'Product')
ax2.set_xlabel('Importance of time (as a product)')
ax2.set_ylabel('Optimal adhesivity')
ax2.set_title('Product of time and retained particles')
plt.tight_layout()
plt.show()