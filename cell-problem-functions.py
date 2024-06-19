import numpy    
from scipy import linalg
def get_cell_problem(cond_4:numpy.ndarray)->tuple:
        """Get the left hand side and right hand side of the cell problem.
        Note that the cell problem is a linear equation of the form Ax=b.

        Parameters
        ----------
        cond_4 : numpy.ndarray
            Conductance of edges in the cell. 
            Note that cond_4[i,j,r0,r1] is the conductance of the edge 
            between node i and node j in the cell at position r0, r1 relative to the cell containing node i.

        Returns
        -------
        (lhs_cpro_2, rhs_cpro_3) : tuple(numpy.ndarray, numpy.ndarray)
            Left hand side and right hand side of cell problem. 
            Note that lhs_cpro_2[i,j] is the element of the left hand matrix for nodes i and j, 
            and rhs_cpro_3[i,j,m] is the element of the right hand matrix for nodes i and j in direction m.
        """
        
        
        # Define readable parameters 
        # -----
        N = 4
        R = 3
        D = 1
        
        refs_1 = numpy.array([-1,0,1])
        leng_1 = numpy.array([1])
        

        # Define arrays to fill
        # -----
        rhs_cpro_inte_5 = numpy.empty(shape=(N,N,R,R,D))
        # rhs_cpro_inte_5[i,j,r0,r1,m].
        
        lhs_inte_4 = numpy.empty(shape=(N,N,R,R))
        # lhs_inte_4[i,j,r0,r1].


        # Build lhs and rhs
        # ------
        for r0 in range(R):
            for r1 in range(R):

                # Get lhs integrand
                # -----
                lhs_inte_4[:,:,r0,r1] = cond_4[:,:,r0,r1] - numpy.diag(numpy.sum(a=cond_4[:,:,r0,r1], axis=1))

                # Get rhs integrand
                # -----                     
                for m in range(D):
                    if m==0: 
                        r=r0
                    elif m==1:
                        r=r1
                    else: 
                        raise Exception("m != 0,1. This is impossible, since num_dims={}".format(D))

                    rhs_cpro_inte_5[:,:,r0,r1,m] = cond_4[:,:,r0,r1]*refs_1[r]*leng_1[m]


        # Sum over references
        # -----
        rhs_cpro_3 = -numpy.sum(a=numpy.sum(a=rhs_cpro_inte_5, axis=3), axis=2) # sum over r1 then r0
        # NB: rhs of cell problem has minus sign by definition.

        lhs_cpro_2 =  numpy.sum(a=numpy.sum(a=lhs_inte_4, axis=3), axis=2) # sum over r1 then r0


        # Force a unique solution 
        # -------
        #lhs_cpro_2[-1,:] = numpy.zeros(num_nodes)
        #lhs_cpro_2[-1,-1] = 1
        #rhs_cpro_3[-1,:,0] = numpy.zeros(num_nodes)
        #rhs_cpro_3[-1,:,1] = numpy.zeros(num_nodes)

        return (lhs_cpro_2, rhs_cpro_3)


def step_cell_problem( lhs_cpro_2:numpy.ndarray, rhs_cpro_3:numpy.ndarray)->numpy.ndarray:
        """Solve the cell problem.
        The cell problem is a linear equation of the form Ax=b.

        Parameters
        ----------
        lhs_cpro_2 : numpy.ndarray
            Left hand side of the cell problem.
            Note that lhs_cpro_2[i,j] is the element of the left hand matrix for nodes i and j.
        rhs_cpro_3 : numpy.ndarray
            Right hand side of cell problem. 
            Note that rhs_cpro_3[i,j,m] is the element of the right hand matrix for nodes i and j 
            in direction m.
        
        Returns
        -------
        csol_2 : numpy.ndarray
            Cell solution. 
            Note that csol_2[i,m] is the cell solution at node i in direction m.
        """

        # Define readable parameters 
        # -----
        N = 4
        D = 1


        # Define arrays to fill
        # -----
        csol_2 = numpy.empty(shape=(N,D))
        # csol_2[i,m]


        # Get solution
        # -----
        a_2 = lhs_cpro_2[:,:]
        for m in range(D):
            b_1 = numpy.sum(a=rhs_cpro_3[:,:,m], axis=1) # sum over j
            csol_2[:,m] = linalg.solve(a_2, b_1)

            # TODO: Consider -- 
            #sol = optimize.lsq_linear(A=a_2,b=b_1)
            #csol_3[k,:,m] = sol.x

        return csol_2

G_initial = {
        (0, 2, 1, 2): 1, (2, 0, 1, 0): 1, (1, 3, 1, 2): 1, (3, 1, 1, 0): 1,
        (0, 1, 0, 1): 1, (1, 0, 2, 1): 1, (2, 3, 0, 1): 1, (3, 2, 2, 1): 1,
        (0, 1, 1, 1): 1, (1, 0, 1, 1): 1, (0, 2, 1, 1): 1, (2, 0, 1, 1): 1,
        (1, 3, 1, 1): 1, (3, 1, 1, 1): 1, (2, 3, 1, 1): 1, (3, 2, 1, 1): 1
}

def initial_G(initial_G_dict):
    """
    Solves for initial G given a dictionary containing non-zero values

    Parameters:
    initial_G_dict (dict): dictionary containing (i,j,r,s): value

    Returns:
    G (np.ndarray): Solution fot initial G of shape (4,4,3,3)
    """
    G = numpy.zeros((4,4,3,3))
    positions = initial_G_dict
    # Assign non-zero values to the tensor
    position= numpy.array(list(positions.keys()))
    values = numpy.array(list(positions.values()))
    idx_i = position[:, 0]
    idx_j = position[:, 1]
    idx_r = position[:, 2]
    idx_s = position[:, 3]
    G[idx_i, idx_j, idx_r, idx_s] = values
    return G
cond_4 = initial_G(G_initial)
# def get_cell_problem(self, cond_4:numpy.ndarray)->tuple:
(lhs_cpro_2, rhs_cpro_3) = get_cell_problem(cond_4 )
print(lhs_cpro_2)
print(f'rhs is {numpy.sum(a = rhs_cpro_3,axis = 1)}')
#def step_cell_problem( lhs_cpro_2:numpy.ndarray, rhs_cpro_3:numpy.ndarray)
csol2 = step_cell_problem(lhs_cpro_2,rhs_cpro_3)
print(csol2)
