from model import initial
import numpy as np
from solver import shape_to_length
import time

def slice_values(concat_array, shape):
    """
    Function that takes the 1d array formed by different flattened tensors and obtains 
    each flattened tenso

    Parameters:
    concat_array (np.ndarray): Array of the concatenated flattened tensors
    shape (tuple): shape of the first tensor of the array

    Returns:
    The 1d array of the first tensor we included
    """
    return 1

def sum_over_last_index(tensor_flat, original_shape):
    num_elements_per_subtensor = original_shape[-1]
    num_subtensors = int(shape_to_length(original_shape)/ num_elements_per_subtensor)
    # Initialize the result array
    result = np.zeros(num_subtensors)
    
    # Sum over s for each subtensor
    for index in range(num_subtensors):
        start_idx = index * num_elements_per_subtensor
        end_idx = start_idx + num_elements_per_subtensor
        result[index] = np.sum(tensor_flat[start_idx:end_idx])
    
    #result = result.reshape((nx, i, j, r))
    return result



x_eval = np.linspace(0,1,40)
tensor, ic = initial(x_eval, (4,4,3,3))
# Sum over s
summed = np.sum(tensor,axis=4)
result = sum_over_last_index(ic, (len(x_eval),4,4,3,3))
summed2 = np.sum(tensor,axis=(4,3,2,1))
result2= sum_over_last_index(result,(len(x_eval),4,4,3))
result3 = sum_over_last_index(result2,(len(x_eval),4,4))
result4 = sum_over_last_index(result3,(len(x_eval),4))
print(np.allclose(summed2.reshape(-1),result4))
print(result4.shape)







def sum_over_axis1_old(tensor_flat, original_shape):
    nx, i, j, r, s = original_shape
    
    # Initialize the result array
    result = np.zeros((nx, j, r, s))
    
    # Sum over axis 1 for each subtensor
    for n in range(nx):
        for jj in range(j):
            for rr in range(r):
                for ss in range(s):
                    sum_value = 0
                    for ii in range(i):
                        index = (n * i * j * r * s) + (ii * j * r * s) + (jj * r * s) + (rr * s) + ss
                        sum_value += tensor_flat[index]
                    result[n, jj, rr, ss] = sum_value

    return result
def sum_over_axis2(tensor_flat,original_shape):
    nx, i, j, r, s = original_shape
    
    # Initialize the result array
    result = np.zeros((nx, i, r, s))
    
    # Sum over axis 1 for each subtensor
    for n in range(nx):
        for ii in range(i):
            for rr in range(r):
                for ss in range(s):
                    sum_value = 0
                    for jj in range(j):
                        index = (n * i * j * r * s) + (ii * j * r * s) + (jj * r * s) + (rr * s) + ss
                        sum_value += tensor_flat[index]
                    result[n, ii, rr, ss] = sum_value

    return result
def sum_over_j(tensor_flat,original_shape):
    nx, i, j, r, s = original_shape
    
    # Initialize the result array
    result = np.zeros((nx, i, r, s))
    total_elements = nx * i * j * r * s
    # Sum over axis 1 for each subtensor
    for idx in range(0, total_elements, i*j*r*s):
        for ii in range(i):
            for rr in range(r):
                for ss in range(s):
                    sum_value = 0
                    for jj in range(j):
                        index = idx + (ii * j * r * s) + (jj * r * s) + (rr * s) + ss
                        sum_value += tensor_flat[index]
                    n = idx // (i * j * r * s)
                    result[n, ii, rr, ss] = sum_value

    return result
def sum_over_i(tensor_flat, original_shape):
    nx, i, j, r, s = original_shape
    result_shape = (nx, j, r, s)
    result = np.zeros(result_shape)
    
    # Calculate the total number of elements in the flattened tensor
    total_elements = nx * i * j * r * s
    
    # Iterate over the flattened tensor with strides
    for idx in range(0, total_elements, i * j * r * s):
        for jj in range(j):
            for rr in range(r):
                for ss in range(s):
                    sum_value = 0
                    for ii in range(i):
                        index = idx + (ii * j * r * s) + (jj * r * s) + (rr * s) + ss
                        sum_value += tensor_flat[index]
                    # Calculate the corresponding indices in the result tensor
                    n = idx // (i * j * r * s)
                    result[n, jj, rr, ss] = sum_value

    return result
