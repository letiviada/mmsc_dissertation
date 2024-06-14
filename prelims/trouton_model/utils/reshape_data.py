import numpy as np

def reshape(data):
    """
    Custom reshape function that converts a list of lists to a transposed NumPy array.

    Args:
        data (list of lists): The input data to be reshaped.

    Returns:
        np.ndarray: The reshaped NumPy array.
    """
    array = np.array(data)
    return np.transpose(array)
