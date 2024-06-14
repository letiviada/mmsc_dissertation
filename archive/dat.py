import numpy as np
import h5py

# Step 1: Generate and store the data in a dictionary
num_s = 20  # Number of s values

# Initialize data storage
data_store = {}

for s in range(num_s):
    tensor = np.random.rand(4, 4, 3, 3)
    array = np.random.rand(4)
    k = np.random.randint(0, 10)
    j = np.random.randint(0, 10)
    
    # Store in the dictionary
    data_store[s] = {
        'tensor': tensor,
        'array': array,
        'k': k,
        'j': j
    }

# Step 2: Save the data to an HDF5 file
with h5py.File('data_store.h5', 'w') as f:
    for s in data_store:
        grp = f.create_group(str(s))
        grp.create_dataset('tensor', data=data_store[s]['tensor'])
        grp.create_dataset('array', data=data_store[s]['array'])
        grp.attrs['k'] = data_store[s]['k']
        grp.attrs['j'] = data_store[s]['j']

# Function to retrieve data from the HDF5 file
def retrieve_data(filename, s):
    with h5py.File(filename, 'r') as f:
        if str(s) in f:
            grp = f[str(s)]
            tensor = np.array(grp['tensor'])
            array = np.array(grp['array'])
            k = grp.attrs['k']
            j = grp.attrs['j']
            return tensor, array, k, j
        else:
            return None, None, None, None

# Function to retrieve all k, j, and corresponding s values
def retrieve_all_kj_s(filename):
    k_values = []
    j_values = []
    s_values = []
    with h5py.File(filename, 'r') as f:
        for s in f.keys():
            grp = f[s]
            k_values.append(grp.attrs['k'])
            j_values.append(grp.attrs['j'])
            s_values.append(float(s)) 
    return s_values, k_values, j_values

# Example usage
s_values, k_values, j_values = retrieve_all_kj_s('data_store.h5')
print(f"All s values: {s_values}")
print(f"All k values: {k_values}")
print(f"All j values: {j_values}")

