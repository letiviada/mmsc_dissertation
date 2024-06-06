import numpy as np
import time
from matplotlib import colormaps

def initial_broadcast(nx, t, values_dict):
    x = np.linspace(0, 1, nx)

    def func(xi, t, value):
        c = 2 / value
        return 2 / (t + xi + c)

    tensor = np.zeros((nx, 9, 4, 4))

    positions = np.array(list(values_dict.keys()))
    values = np.array(list(values_dict.values()))

    idx_nx = np.arange(nx)[:, None]
    idx_r = positions[:, 0]
    idx_i = positions[:, 1]
    idx_j = positions[:, 2]

    xi = x[:, None]
    c = 2 / values
    tensor_values = func(xi, t, values)

    tensor[idx_nx, idx_r, idx_i, idx_j] = tensor_values

    initial_condition = tensor.reshape(-1)
    return tensor, initial_condition

# Example usage
nx = 10000
t = 1

values_dict = {
    (1, 0, 2): 1, (1, 1, 3): 2,
    (3, 0, 1): 3, (3, 2, 3): 4,
    (4, 0, 1): 5, (4, 0, 2): 6, (4, 1, 0): 5, (4, 1, 3): 7, (4, 2, 0): 6, (4, 2, 3): 8, (4, 3, 1): 7, (4, 3, 2): 8,
    (5, 1, 0): 9, (5, 3, 2): 10,
    (7, 2, 0): 11, (7, 3, 1): 12
}

start_time = time.time()
tensor, initial_condition = initial_broadcast(nx=nx, t=t, values_dict=values_dict)
print(f"Broadcasting time: {time.time() - start_time} seconds")

def initial_loop(nx, t, values_dict):
    x = np.linspace(0, 1, nx)

    def func(xi, t, value):
        c = 2 / value
        return 2 / (t + xi + c)

    tensor = np.zeros((nx, 9, 4, 4))

    for i, xi in enumerate(x):
        for pos, val in values_dict.items():
            tensor[i, pos[0], pos[1], pos[2]] = func(xi, t, val)

    initial_condition = tensor.reshape(-1)
    return tensor, initial_condition

# Example usage
start_time = time.time()
tensor, initial_condition = initial_loop(nx=nx, t=t, values_dict=values_dict)
print(f"For loop time: {time.time() - start_time} seconds")

print(list(colormaps))
