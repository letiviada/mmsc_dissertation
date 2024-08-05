import numpy as np
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
from regression.utils_r import get_data_from_json
from scipy.stats.qmc import LatinHypercube, scale
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv('/Users/letiviada/dissertation_mmsc/regression/optimization/opt_throughput/data/throughput_100/data_varying_n_min.csv')

df = data[data['n'] == 1]
df_ps = df[df['particle_size'] == 0.06]

df_new = df_ps[['adhesivity', 'ratio']]

sns.scatterplot(data=df_new, y='adhesivity', x='ratio')
plt.ylabel('Adhesivity')
plt.xlabel('Ratio')
plt.title('Scatter Plot of Adhesivity vs Ratio')
plt.show()

X = df_new[['ratio']].values
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
df_new['cluster'] = clusters

plt.scatter(df_new['ratio'], df_new['adhesivity'], c=df_new['cluster'], cmap='viridis')
plt.show()

