import numpy as np
from casadi import *
# Define G as a dictionary
G_dict = {
    (0, 2, 1, 2): 1, (2, 0, 1, 0): 1, 
    (1, 3, 1, 2): 2, (3, 1, 1, 0): 2, 
    (0, 1, 0, 1): 3, (1, 0, 2, 1): 3, 
    (2, 3, 0, 1): 4, (3, 2, 2, 1): 4, 
    (0, 1, 1, 1): 5, (1, 0, 1, 1): 5, 
    (0, 2, 1, 1): 6, (2, 0, 1, 1): 6, 
    (1, 3, 1, 1): 7, (3, 1, 1, 1): 7, 
    (2, 3, 1, 1): 8, (3, 2, 1, 1): 8
}

# Initialize G array
G_val = np.zeros((4, 4, 3, 3))

# Populate G array using the dictionary
for key, value in G_dict.items():
    G_val[key] = value

# Define Delta and H as ones
Delta_val = np.ones((4, 4, 3,3))
H_val = np.ones((4, 4, 3,3))
alpha_val = 1.0

tensor = G_val * Delta_val * H_val
# Sum over the specified axes
k_numpy = 0.5 * np.sum(tensor,axis = (3,2,1,0))
j_numpy = -np.sum(tensor, axis = (3,2,1,0))

print(f"Numpy k: {k_numpy}, Numpy j: {j_numpy}")
# Flatten the tensor
G_flat = G_val.flatten()
Delta_flat = Delta_val.flatten()
H_flat = H_val.flatten()

tensor.flat = tensor.flatten()

# Define the flattened tensors as symbolic variables in CasADi
G_sym = SX.sym('G', G_flat.size)
Delta_sym = SX.sym('Delta', G_flat.size)
H_sym = SX.sym('H', G_flat.size)
alpha_sym = SX.sym('alpha')

alpha_val = 1.0

def get_k(G, Delta):
    k = 0
    for i in range(G.size()[0]):
        k += 1 * G[i] * Delta[i]
    return 0.5 * k

def get_j(G, Delta, H, alpha):
    j = 0
    for i in range(G.size()[0]):
        j += alpha * G[i] * Delta[i] * H[i]
    return -j

# Create CasADi functions
f_k = Function('f_k', [G_sym, Delta_sym], [get_k(G_sym, Delta_sym)])
f_j = Function('f_j', [G_sym, Delta_sym, H_sym, alpha_sym], [get_j(G_sym, Delta_sym, H_sym, alpha_sym)])

# Evaluate the functions with the given values
k_val = f_k(G_flat, Delta_flat)
j_val = f_j(G_flat, Delta_flat, H_flat, alpha_val)

print(f"CasADi k: {k_val}, CasADi j: {j_val}")






