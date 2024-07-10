from utils import get_data_from_json, data_time
import matplotlib.pyplot as plt

ml_data = get_data_from_json('performance_indicators/performance_indicators_phi_1.0.json')
data = data_time(400, ['Volume Liquid'], ml_data)

# Plot each row
plt.figure(figsize=(10, 6))
for i, row in data.iterrows():
    if row['Volume Liquid_time_400'] == row['Lifetime']:
        pass
        #plt.scatter(row['time'][-1],row['Last Concentration'], marker = 'x', color = 'red')
        #plt.plot(row['time'], row['Throughput'], label=f'Row {i}')
    else:
        plt.scatter(400,row['Volume Liquid_time_400'], marker = 'x', color = 'red')
        plt.plot(row['time'], row['Throughput'], label=f'Row {i}')

# Adding title and labels
plt.title('Throughput vs. Time')
plt.xlabel('Time')
plt.ylabel('Throughput')
plt.legend()
plt.grid(True)
plt.show()