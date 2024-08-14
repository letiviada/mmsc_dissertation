import sys
sys.path.append('/Users/letiviada/dissertation_mmsc/regression/')
from utils_r import open_model, get_ratio, save_data_to_csv
from models import train_model
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

#mse, r2 = train_model(output, data, size_train = size, type_model=type_model,size_sampling = sampling_method,save = False)
data = pd.read_csv('regression/optimization/opt_throughput/data/throughput_100/data_for_sums.csv')
true_sol = pd.read_csv('regression/optimization/opt_throughput/data/throughput_100/data_varying_n_min.csv')

def open_models(name_model:str, type_model:str = 'polynomial')->tuple:
    """
    Function that opens the models

    Parameters:
    -----------
    name_model (str): the name of the model to be used
    model_path (str): the path to the model

    Returns:
    --------
    ml_model (object): the model
    """
    model_path = f'regression/models_{type_model}/'
    ml_model = open_model(name_model,model_path=model_path)
    return ml_model

def train_and_open(train:bool,data:pd.DataFrame)->tuple:
    """
    Function that trains both models
    """
    data_particles = data.drop(columns = ['time_throughput_100'])
    data_time = data.drop(columns = ['efficiency_throughput_100'])
    if train == True:
        train_model('efficiency_throughput_100', data_particles, size_train = 'all', type_model = 'polynomial', save = True)
        train_model('time_throughput_100', data_time, size_train = 'all', type_model = 'polynomial', save = True)
    ml_particles = open_models('efficiency_throughput_100')
    ml_time = open_models('time_throughput_100')
    return ml_particles, ml_time
ml_particles, ml_time = train_and_open(train = False,data=data)

def train_ratio(train:bool, data:pd.DataFrame, n:int)->pd.DataFrame:
    """
    Function that trains the ratio model
    """
    data_model = data[['adhesivity','particle_size', 'ratio']]
    if train == True:
        train_model('ratio', data_model, size_train = 'all', type_model = 'polynomial', save = True)
    ml_ratio = open_models('ratio')
    return ml_ratio
ps = 0.05
n = 3

adhesivity_values = np.linspace(0, 0.75, num=101)
#particle_size_values = np.linspace(0.01, 0.1, num=10).round(3)
particle_size_values = ps * np.ones(len(adhesivity_values))

# Flatten the meshgrid and create a DataFrame
df = pd.DataFrame({
    'adhesivity': adhesivity_values,
    'particle_size': particle_size_values
})

# Predict the retained particles and time
df['efficiency_particles'] = ml_particles.predict(df)
df['time'] = ml_time.predict(df[['adhesivity', 'particle_size']])
data_ratio = get_ratio('efficiency_particles','time',n,df)

# Filter the DataFrame for particle size = 0.06
df_filtered = data_ratio[data_ratio['particle_size'] == ps]
df_true = true_sol[true_sol['particle_size'] == ps]
df_true_n = df_true[df_true['n'] == n]

plt.figure()
sns.scatterplot(data=df_filtered, x='adhesivity', y='ratio')
sns.scatterplot(data=df_true_n, x='adhesivity', y='ratio', color='red')
plt.show()


