import matplotlib.pyplot as plt
import numpy as np

G = -4.0 * np.ones(144).reshape((4,4,3,3))
print(G)
G = np.abs(G)
print(G)

