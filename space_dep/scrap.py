from data.N4 import values_dict
print(values_dict.keys())
cells= [(1,3,4,5),(1,3,0,0)]
for idx, cell in enumerate(values_dict.keys()):
    print(idx, cell)