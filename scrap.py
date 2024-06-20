import numpy as np
from initial_conditions_2D import four_reg_prescribed
from equations_preprocess import *

refs_1 = np.array([0,1,-1])
leng_1 =  4*np.array([1.0,1.0])
adhe_4 = 1 * np.ones(shape=(4,4,3,3))

cond_4 = four_reg_prescribed(4,3)
(lhs_cpro_2, rhs_cpro_3) = get_cell_problem(cond_4,refs_1,leng_1)
csol_2 = step_cell_problem(lhs_cpro_2, rhs_cpro_3)
delt_4 = get_delta(csol_2, refs_1, leng_1)
heav_4 = get_heaviside(delt_4)
(perm_2, depo_1) = get_permeability_and_adhesivity(adhe_4,cond_4,delt_4,heav_4,refs_1,leng_1)
print(perm_2)
print(depo_1)


