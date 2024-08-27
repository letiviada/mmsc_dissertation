import pandas as pd
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
# Calculate R2 score


df_ml = pd.read_csv('regression/optimization/opt_time/data/ml_range/optimum_values.csv')
df_p = pd.read_csv('regression/optimization/opt_time/data/physical_large/optimum_values.csv')
df_ml2 = df_ml.groupby(['particle_size', 'weight_coefficient'])['adhesivity_time_400'].mean().reset_index()
df_p2 = df_p.groupby(['particle_size', 'weight_coefficient'])['adhesivity_time_400'].mean().reset_index()
df_ml2['Adhesivity_Std_Per_Size'] = df_ml.groupby(['particle_size', 'weight_coefficient'])['adhesivity_time_400'].std().reset_index()['adhesivity_time_400']
df_p2['Adhesivity_Std_Per_Size'] = df_p.groupby(['particle_size', 'weight_coefficient'])['adhesivity_time_400'].std().reset_index()['adhesivity_time_400']
increased_sigma = 2.5 
df_ml2['Adhesivity_Smoothed_Per_Size'] = df_ml2.groupby('particle_size')['adhesivity_time_400'].transform(
    lambda x: gaussian_filter1d(x, sigma=increased_sigma)
)
df_p2['Adhesivity_Smoothed_Per_Size'] = df_p2.groupby('particle_size')['adhesivity_time_400'].transform(
    lambda x: gaussian_filter1d(x, sigma=increased_sigma)
)


fig, axs = plt.subplots(1, 2, figsize=(15, 8))

for i, size in enumerate([0.04, 0.08]):
    data_part_size_ml = df_ml2[df_ml2['particle_size'] == size]
    data_part_size_p = df_p2[df_p2['particle_size'] == size]
    #r2_score_value = r2_score(df2_agg['adhesivity_time_400'], df_agg['adhesivity_time_400'])
    #print(f'R2 score for size {size}: {r2_score_value}')
    colors = ['blue', 'orange', 'green', 'red']
    sns.lineplot(data = data_part_size_p, x = 'weight_coefficient', y = 'adhesivity_time_400',
                   color = colors[0], ax = axs[i])
    sns.lineplot(data = data_part_size_ml, x = 'weight_coefficient', y = 'adhesivity_time_400', color = colors[1], ax = axs[i])
    
plt.xlabel('n')
plt.ylabel('Adhesivity')
plt.tight_layout()
plt.show()

