
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

filename = f'regression/opt_throughput/data/throughput_100/data_for_sums.csv'
data = pd.read_csv(filename)


df1 = data[data['particle_size'] == 0.06]
df = pd.DataFrame({
    'Adhesivity': df1['adhesivity'],
    'Time': df1['time_throughput_100'],
    'Retained': df1['avg_retained_particles_throughput_100']

})
scaler = MinMaxScaler()
df[['Time', 'Retained']] = scaler.fit_transform(df[['Time', 'Retained']])
print(df)

def calculate_score(time, retained, time_weight=0.5):
    return time_weight * time + (1-time_weight) * retained

# Apply the metric to each row
df['Score'] = df.apply(lambda row: calculate_score(row['Time'], row['Retained'], time_weight=1), axis=1)

# Find the row with the minimum score
optimal_solution = df.loc[df['Score'].idxmin()]
print(optimal_solution)

# Sweep over weight combinations
weight_combinations =  np.linspace(0, 1, 1001)
results = []

for time_weight in weight_combinations:
    df['Score'] = df.apply(lambda row: calculate_score(row['Time'], row['Retained'], time_weight), axis=1)
    optimal_solution = df.loc[df['Score'].idxmin()]
    results.append((time_weight, optimal_solution['Adhesivity'],optimal_solution['Time'], optimal_solution['Retained'], optimal_solution['Score']))
# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results, columns=['n', 'adhesivity', 'Optimal Time', 'Optimal Retained', 'Optimal Score'])
results_df['particle_size'] = 0.06
results_df.to_csv('regression/opt_throughput/data/throughput_100/results_sum.csv', index=False)


