#from data.helpers import map_indices, create_value_dict
from helpers import map_indices, create_value_dict
# Defining the points in the form of (i,j,r,s) we do (i-1,j-1,r+1,s+1)
ic_dict_math = {
    (1,3,0,1): 1, (3,1,0,-1): 1,
    (2,4,0,1): 2, (4,2,0,-1): 2,
    (1,2,-1,0): 3, (2,1,1,0): 3,
    (3,4,-1,0): 4, (4,3,1,0): 4,
    (1,2,0,0): 5, (2,1,0,0): 5,
    (1,3,0,0): 6, (3,1,0,0): 6,
    (2,4,0,0): 7, (4,2,0,0): 7,
    (3,4,0,0): 8, (4,3,0,0): 8,

}
values_dict_math = create_value_dict(ic_dict_math)
# Outputs that we need
ic_dict = {map_indices(key): value for key, value in ic_dict_math.items()}
values_dict = {map_indices(key): value for key, value in values_dict_math.items()}