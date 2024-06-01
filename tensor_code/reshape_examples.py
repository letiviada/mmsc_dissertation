import numpy as np

x = np.arange(16*9)
print(f'The original array is {x}')

x_tensor = x.reshape(9,4,4)

print(f'The tensor array is {x_tensor}')

#y_tensor = x.reshape(3,4,4)
#print(f'The tensor array is {y_tensor}')
#print(y_tensor.shape)

# There are three ways of reshaping a tensor into a 1D array, it is done by rows. 
#y1 = y_tensor.reshape(-1)
#y2 = y_tensor.flatten()
#y3 = y_tensor.ravel()








