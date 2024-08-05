import numpy as np
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
import pandas as pd
from collections import defaultdict


data = pd.read_csv('/Users/letiviada/dissertation_mmsc/regression/optimization/opt_throughput/data/throughput_100/data_varying_n_min.csv')

df = data[data['n'] == 1]

df_new = df[['adhesivity', 'particle_size']]
df_part = df_new[df_new['particle_size'] == 0.01]
print(df_new)

point_counts = defaultdict(int)
for _, row in df_new.iterrows():
            point = tuple(row)
            point_counts[point] += 1
for _, row in df_part.iterrows():
            point = tuple(row)
            point_counts[point] += 1

point_counts_df = pd.DataFrame(list(point_counts.items()), columns=['Point', 'Count'])
print(point_counts_df)
