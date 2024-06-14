import numpy as np
#from model import initial

def sum_over_s_r_j_i(tensor_flat, original_shape):
    nx, i, j, r, s = original_shape

    # Step 1: Sum over axis s
    sum_s_shape = (nx, i, j, r)
    sum_s_result = np.zeros(sum_s_shape)
    
    # Iterate and sum over s
    for n in range(nx):
        for ii in range(i):
            for jj in range(j):
                for rr in range(r):
                    sum_value = 0
                    for ss in range(s):
                        index = (n * i * j * r * s) + (ii * j * r * s) + (jj * r * s) + (rr * s) + ss
                        sum_value += tensor_flat[index]
                    sum_s_result[n, ii, jj, rr] = sum_value

    # Step 2: Sum over axis r
    sum_r_shape = (nx, i, j)
    sum_r_result = np.zeros(sum_r_shape)
    
    # Iterate and sum over r
    for n in range(nx):
        for ii in range(i):
            for jj in range(j):
                sum_value = 0
                for rr in range(r):
                    sum_value += sum_s_result[n, ii, jj, rr]
                sum_r_result[n, ii, jj] = sum_value

    # Step 3: Sum over axis j
    sum_j_shape = (nx, i)
    sum_j_result = np.zeros(sum_j_shape)
    
    # Iterate and sum over j
    for n in range(nx):
        for ii in range(i):
            sum_value = 0
            for jj in range(j):
                sum_value += sum_r_result[n, ii, jj]
            sum_j_result[n, ii] = sum_value

    # Step 4: Sum over axis i
    sum_i_shape = (nx,)
    sum_i_result = np.zeros(sum_i_shape)
    
    # Iterate and sum over i
    for n in range(nx):
        sum_value = 0
        for ii in range(i):
            sum_value += sum_j_result[n, ii]
        sum_i_result[n] = sum_value

    return sum_i_result

# Example usage
#original_shape = (100, 5, 6, 7, 8)
#tensor = np.random.rand(*original_shape)

#x_eval = np.linspace(0,1,40)
#shape = (4,4,3,3)
#tensor, ic = initial(x_eval, shape)
# Sum using defined function
#result_custom = sum_over_s_r_j_i(ic, (len(x_eval),*shape))

# Sum over s, then r, then j, and then i using np.sum
#result_np = np.sum(tensor, axis=(4, 3, 2, 1))

# Verify the results
#print("Custom function result shape:", result_custom.shape) 
#print("NumPy function result shape:", result_np.shape)       
#print("Results match:", np.allclose(result_custom, result_np))
