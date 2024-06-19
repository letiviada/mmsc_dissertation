from initial_conditions_2D import four_reg_prescribed
from equations_preprocess import get_cell_problem, step_cell_problem, get_permeability_and_adhesivity
import numpy
G = four_reg_prescribed(4,3)
(lhs_cpro_2, rhs_cpro_3) = get_cell_problem(G)
print(lhs_cpro_2)
print(rhs_cpro_3)
csol_2 = step_cell_problem(lhs_cpro_2, rhs_cpro_3)
print(csol_2)


#def get_permeability_and_adhesivity(adhe_4:numpy.ndarray, cond_4:numpy.ndarray, 
                                        #delt_4:numpy.ndarray, heav_4:numpy.ndarray, 
                                        #refs_1:numpy.ndarray, leng_1:numpy.ndarray)->tuple:

adhe_4 = 1.0 * numpy.ones(shape = (4,4,3,3))
cond_4 = G

(perm, depo) = get_permeability_and_adhesivity()