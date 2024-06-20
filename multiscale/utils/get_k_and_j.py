import numpy as np

# Assuming s[t, x] is a 2D numpy array and k_s is a function or interpolator
def get_k_and_j(s_tx, k_s):
    # Initialize k[t, x] with the same shape as s[t, x]
    k_tx = np.zeros_like(s_tx)
    
    # Iterate over all t, x indices
    for t in range(s_tx.shape[0]):
        for x in range(s_tx.shape[1]):
            # Find the corresponding s value
            s_value = s_tx[t, x]
            # Get k value using the k[s] function/interpolator
            k_value = k_s(s_value)
            # Assign the k value to k[t, x]
            k_tx[t, x] = k_value
    
    return k_tx