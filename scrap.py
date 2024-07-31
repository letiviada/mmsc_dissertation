import numpy as np
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
from regression.utils import get_data_from_json
from sklearn.neighbors import NearestNeighbors
from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from scipy.stats.qmc import LatinHypercube, scale
import matplotlib.pyplot as plt 
import pandas as pd

data = get_data_from_json('performance_indicators/performance_indicators_sample_size.json')
print(data.shape)
data_lhs = data[['adhesivity','particle_size', 'termination_time']]
lhs = LatinHypercubeSampling(data_input = data_lhs,number_of_samples = 40, sampling_type = 'selection', xlabels = ['adhesivity','particle_size'], ylabels = ['termination_time'])
inputs_random, outputs_random = data_lhs.drop(columns='termination_time'), data_lhs['termination_time']  
random_samples = inputs_random.sample(n=40)
outputs_samples = outputs_random.loc[random_samples.index]

def get_lhs_unique_points(data, size):
    unique_points = pd.DataFrame()
    
    while len(unique_points) < size:
        # Define the Latin Hypercube sampler
        lhs_scipy = LatinHypercube(d=2)
        sample = lhs_scipy.random(n=size * 2)  # Generate more points than needed to ensure we get enough unique points
        l_bounds = np.array([data['adhesivity'].min(), data['particle_size'].min()])
        u_bounds = np.array([data['adhesivity'].max(), data['particle_size'].max()])
        sample_scaled = scale(sample, l_bounds, u_bounds)
        
        # Create a DataFrame from the scaled sample
        sample_df = pd.DataFrame(sample_scaled, columns=['adhesivity', 'particle_size'])
        
        # Find the nearest neighbors in the original dataset
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data[['adhesivity', 'particle_size']])
        distances, indices = nbrs.kneighbors(sample_df)
        closest_points = data.iloc[indices.flatten()]
        
        # Combine and ensure the points are unique
        unique_points = pd.concat([unique_points, closest_points]).drop_duplicates(subset=['adhesivity', 'particle_size'])
    
    return unique_points.head(size)
closest_points = get_lhs_unique_points(data_lhs, 220)
#sample_points = lhs.sample_points()
#plt.scatter(sample_points['adhesivity'], sample_points['particle_size'], marker='x')
plt.scatter(closest_points['adhesivity'], closest_points['particle_size'], color='red')
plt.xlabel('Adhesivity')
plt.ylabel('Particle Size')
plt.title('Latin Hypercube Sampling')
plt.show()


