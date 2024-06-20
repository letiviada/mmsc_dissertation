import numpy as np
import scipy.sparse.linalg as linalg

def solve_W(G: np.ndarray,l: float) -> np.ndarray:
        """
        W is a vector that holds information about the cell solution.

        Parameters:
        ----------
        G (np.ndarray): Input array of shape (N,N,R,R).
        l (int): Length of the filtre

        Returns:
        ----------
        W (np.ndarray): Output array of shape (N,), cell solution.
        """
        # Define parameters
        # -----
        N,R = 4,3
        refs_1 = np.array([0,1,-1]) # r array for the reference set
        leng_1 = l * np.array([1.0]) # length of the filter

        # Build LHS
        # ------
        G_summed = np.sum(G, axis=(3,2)) # Sum over r,s
        Gk_kronecker = np.diag(np.sum(G_summed, axis=1)) # Sum over j (diagonal matrix)
        LHS = G_summed - Gk_kronecker # LHS of the cell problem

        # Build rhs
        # ------
        rhs_5 = np.empty(shape=(N,N,R,R)) # rhs of the cell problem
        for r0 in range(R):
            for s in range(R):

                rhs_5[:,:,r0,s] = G[:,:,r0,s]*refs_1[r0]*leng_1[0]
        rhs_cpro_3 = -np.sum(a=np.sum(a=rhs_5, axis=3), axis=2) # sum over r1 then r0

        # Get solution
        # -----
        RHS_1 = np.sum(a=rhs_cpro_3[:,:], axis=1) # sum over j
        W = linalg.lsqr(A=LHS, b=RHS_1)[0]
        return W

def find_delta(W: np.ndarray,l:float) -> np.ndarray:
    """
    Delta is a tensor that holds information about the difference between cell solutions
    at different nodes.

    Parameters:
    -----------
    W (np.ndarray): Input array of shape (N,).
    l (float): Length of the filtre (scalar).

    Returns:
    --------
    Delta (np.ndarray): Output array of shape (N, N, R), 
    component of the pressure difference in edge ijr per unit pressure gradient
    """
    # Define parameters
    # -----
    N,R = 4,3
    delta = np.empty(shape=(N,N,R))
    delt_4 = np.empty(shape=(N,N,R))
    refs_1 = np.array([0,1,-1])
    leng_1 = l * np.array([1.0])
        
    # Fill using definition of delta
    # -----
    for i in range(N):
        for j in range(N):
            for r in range(R):
                delt_4[i,j,r] = W[i] - (W[j] + refs_1[r]*leng_1[0])

    W_matrix_2 = W[:, np.newaxis] - W # W_i - W_j
    rl_array = np.array([0, 1,-1])*l # r*l
    for r in range(R):
        delta[:,:,r] = W_matrix_2[:,:] -rl_array[r] # W_i - W_j - r*l
    return delt_4

def find_permeability(G: np.ndarray,delta: np.ndarray) -> float:
    """
    k is a parameter that describes the permeability of the cell.

    Parameters:
    G (np.ndarray): Pore conductance (N,N,R,R).
    Delta (np.ndarray): Pressure difference (N,N,R)

    Returns:
    k (float):  The permeability, which is the effective conductance.
    """
    # Define the parameters
    # -----
    N, R = 4,3
    D = 1
    delta_r = np.empty((N,N,R)) # delta_r[i,j,r]
    r_arr = np.array([0,1,-1]) # r array for the reference set
    leng_1 = np.array([1.0]) # length of the filter
    # Multiply delta by r
    for idx_r in range(R):
        delta_r[:,:,idx_r] = delta[:,:,idx_r] * r_arr[idx_r]
    # Make delta match the shape of G
    delta_4 = np.repeat(a=delta_r[:,:,:,np.newaxis],repeats=3,axis=-1)
    # Compute the integrand
    integrand = - G * delta_4
    # Compute k
    k = 0.5 * np.sum(integrand) # Sum over r,s,j,i
         # Make arrays to fill
        # -----
    perm_inte_6 = np.empty(shape=(N,N,R,R,D,D))
        # perm_inte_7[i,j,r0,r1,m,n]

    depo_inte_5 = np.empty(shape=(N,N,R,R,D))
        # depo_inte_6[i,j,r0,r1,m]
    for m in range(D):
        for n in range(D):
            for r0 in range(R):
                for r1 in range(R):
                    if m==0: 
                        rm=r0
                    elif m==1:
                        rm=r1
                    else: 
                        raise Exception("m != 0,1. This is impossible, since the problem is 2D.")
                    if n==0: 
                        rn=r0
                    elif n==1:
                        rn=r1
                    else: 
                        raise Exception("n != 0,1. This is impossible, since the problem is 2D.")
                    # Get depo and perm
                    # -----
                    perm_inte_6[:,:,r0,r1,m,n] = r_arr[rm]*G[:,:,r0,r1]*(-delta[:,:,rn])
                    # TODO: Check heav definition and indexing

        # Sums
        # -----
        perm_5 = np.sum(a=perm_inte_6, axis=3) # sum over r1
        perm_4 = np.sum(a=perm_5, axis=2) # sum over r0
        perm_3 = np.sum(a=perm_4, axis=1) # sum over j
        perm_2 = np.sum(a=perm_3, axis=0) # sum over i
        # perm_2[m,n]    
        # depo_2[m]


        # Multiply by prefactors
        # -----
        for m in range(D):
            for n in range(D):
                perm_2[m,n] = 0.5*(leng_1[m]/np.prod(leng_1))*perm_2[m,n]

    #print(f'k is {k}, and perm_2 is {perm_2}, {k == perm_2[0,0]}')
    return k

def get_heaviside(delt_4:np.ndarray)->np.ndarray:
        """
        Get the Heaviside function. 
        Heaviside is a tensor that holds information about the sign of the difference between cell solutions
        at different nodes. This is positive only when the difference tensor indicates 
        that information is conducted in the direction of the pressure gradient. 
        Note that the tolerance is arbitrarily set at 1E-5 so delt_4 values smaller than this 
        result in zero contribution to adhesivity.

        Parameters
        ----------
        delt_4 : numpy.ndarray
            The cell solution difference.
            Note that delt_4[i,j,r,m] is difference in cell solutions at nodes i and j 
            in direction m with reference r.
            

        Returns
        -------
        heav_4 : numpy.ndarray
            The cell solution difference sign. An indicator for flow along an edge with a component 
            in the direction of the pressure gradient.
            Note that heav_4[i,j,r,m] == 1 if the difference in cell solutions at nodes i and j 
            in direction m with reference r is positive (i.e., above the tolerance), and zero otherwise. 
        """
        tol = 1E-5
        heav_4 = (delt_4>tol).astype(int)
        return heav_4

def compute_heaviside(x: np.ndarray, tolerance: float =1e-5) -> np.ndarray:
    """
    Custom Heaviside step function with tolerance.

    Parameters:
    -----------
    x (np.ndarray or float): Input value(s).
    tolerance (float): Tolerance level for considering values close to zero as zero.

    Returns:
    --------
    np.ndarray: Heaviside step function result.
    """
    # Apply the Heaviside function with tolerance
    x = np.asarray(x)
    result = np.where(x > tolerance, 1, 0)
    return result

def find_adhesivity(alpha: float,G: np.ndarray,delta: np.ndarray,l: float) -> float:
    """
    j is a parameter that describes the adhesivity of the cell.

    Parameters:
    -----------
    alpha (float): Stickiness
    G (np.ndarray): Pore conductance shape (N,N,R,R)
    Delta (np.ndarray): Pressure difference shape (N,N,R)
    l (float): Length of the filtre

    Returns:
    --------
    j (float):  The adhesivity, which is the effective adherence.
    """    

    # Make delta match the shape of G
    delta_4 = np.repeat(a=delta[:,:,:,np.newaxis],repeats=3,axis=-1)
    integrand =  - alpha * G * delta_4 * (1-compute_heaviside(-G * delta_4))
    j = -np.sum(integrand)
        
    N, R = 4,3
    depo_inte_5 = np.empty(shape=(N,N,R,R))
        # depo_inte_5[i,j,r0,r1,m]
    heav_4 = compute_heaviside(-delta)
    adhe_4 = alpha * np.ones(shape=(N,N,R,R))

    # Get adhesivity integrand
    # ------
    for r0 in range(R):
        for r1 in range(R):
            depo_inte_5[:,:,r0,r1] = adhe_4[:,:,r0,r1]*G[:,:,r0,r1]*(-delta[:,:,r0])*(1-compute_heaviside(-G[:,:,r0,r1]*delta[:,:,r0]))
                        # TODO: Check heav definition and indexing
    depo_4 = np.sum(a=depo_inte_5, axis=3) # sum over r1
    depo_3 = np.sum(a=depo_4, axis=2) # sum over r0
    depo_2 = np.sum(a=depo_3, axis=1) # sum over j
    depo_1 = np.sum(a=depo_2, axis=0) # sum over i

    # Multiply by prefactors
    # ----------------------
    depo_1 = -(1/l) * depo_1
    #print(f'j is {j}, and perm_2 is {depo_1}, {j == depo_1}')
    
    return depo_1

def solve_G(alpha:float, beta:float, delta: np.ndarray, G_previous: np.ndarray,tau: np.ndarray) -> np.ndarray:
    """
    Solves for G given W and previous G.

    Parameters:
    alpha (float): Adhesitivity
    beta (float): Particle Size
    delta (np.ndarray): Pressure difference size (N,N,R)
    G_previous (np.ndarray): Previous G array of shape (N,N,R,R).
    tau (np.ndarray): Values of tau.

    Returns:
    G (np.ndarray): Output array of shape (N,N,R,R), conductance.
    """
    # Make delta match the shape of G
    # -----
    delt_4 = np.repeat(a=delta[:,:,:,np.newaxis],repeats=3,axis=-1)

    # Define parameters
    # -----
    dtau = tau[1] - tau[0] / (len(tau)-1)

    # Use definition of G
    # -----
    G_new = G_previous + dtau * (-alpha * beta * (G_previous ** (3/2)) * np.abs(delt_4))
    return G_new

def initial_G(initial_G_dict: dict) -> np.ndarray:
    """
    Solves for initial G given a dictionary containing non-zero values

    Parameters:
    initial_G_dict (dict): dictionary containing (i,j,r,s): value

    Returns:
    G (np.ndarray): Solution fot initial G of shape (N,N,R,R).
    """
    # Define parameters
    # -----
    N,R = 4,3
    G = np.zeros(shape = (N,N,R,R))
    # Get the positions and values
    positions = initial_G_dict
    position= np.array(list(positions.keys()))
    values = np.array(list(positions.values()))
    # Get the indices
    idx_i = position[:, 0]
    idx_j = position[:, 1]
    idx_r = position[:, 2]
    idx_s = position[:, 3]
    # Fill the values
    G[idx_i, idx_j, idx_r, idx_s] = values
    return G

def four_reg_prescribed(num_nodes: int, num_refs: int):
    """
    Make a four-regular cell with manually prescribed edge conductances 
    for num_nodes = 1,4,9, for testing.

    Parameters 
    ----------

    - num_nodes: int
        Number of nodes in the cell.
    - num_refs: int
        Number of lengths in the reference set. 
        For example, if reference set is r=[-1,0,+1] then num_refs==3.
    
    Returns 
    -------
    - cond_init_4: numpy.ndarray 
        cond_init_4[i,j,r,s] is the initial conductance of the edge 
        between node i in one cell and node j in the cell at position r,s relative 
        to the first cell.
    """
    
    # Define parameters 
    # -----
    cond_init_4 = np.zeros(shape=(num_nodes, num_nodes, num_refs, num_refs))
    
    if num_nodes == 1:

        # Grid of one node
        # ----------------
        cond_init_4[0,0,1,0]  = 1
        cond_init_4[0,0,-1,0] = 1

        cond_init_4[0,0,0,1]  = 1
        cond_init_4[0,0,0,-1] = 1


    elif num_nodes == 4:
        
        # Grid of four nodes
        #--------------------
        #           2         3 
        #           |         |
        #         (1.0)     (1.0)
        #           |         |
        # 1--(1.0)--0--(1.0)--1--(1.0)--0
        #           |         |
        #         (1.0)     (1.0)
        #           |         |
        # 3--(1.0)--2--(1.0)--3--(1.0)--2
        #           |         |
        #         (1.0)     (1.0)
        #           |         |
        #           0         1

        # Internal edges
        cond_init_4[0,1,0,0] = 1.0
        cond_init_4[1,0,0,0] = 1.0

        cond_init_4[1,3,0,0] = 1.0
        cond_init_4[3,1,0,0] = 1.0

        cond_init_4[2,3,0,0] = 1.0
        cond_init_4[3,2,0,0] = 1.0

        cond_init_4[0,2,0,0] = 1.0
        cond_init_4[2,0,0,0] = 1.0

        ## External edges
        cond_init_4[1,0,1,0]  = 1.0
        cond_init_4[0,1,-1,0] = 1.0

        cond_init_4[3,2,1,0]  = 1.0
        cond_init_4[2,3,-1,0] = 1.0
        
        cond_init_4[0,2,0,1]  = 1.0
        cond_init_4[2,0,0,-1] = 1.0
        
        cond_init_4[1,3,0,1]  = 1.0
        cond_init_4[3,1,0,-1] = 1.0


    elif num_nodes == 9:
        # Internal edges
        cond_init_4[0,1,0,0] = 1.0#0.8 #1.0
        cond_init_4[1,0,0,0] = 1.0#0.8 #1.0

        cond_init_4[1,2,0,0] = 1.0#0.8 #1.0
        cond_init_4[2,1,0,0] = 1.0#0.8 #1.0

        cond_init_4[3,4,0,0] = 1.0#0.8 #1.0
        cond_init_4[4,3,0,0] = 1.0#0.8 #1.0

        cond_init_4[4,5,0,0] = 1.0#0.8 #1.0
        cond_init_4[5,4,0,0] = 1.0#0.8 #1.0
        
        cond_init_4[6,7,0,0] = 1.0#0.8 #1.0
        cond_init_4[7,6,0,0] = 1.0#0.8 #1.0

        cond_init_4[7,8,0,0] = 1.0#0.8 #1.0
        cond_init_4[8,7,0,0] = 1.0#0.8 #1.0

        cond_init_4[0,3,0,0] = 1.0#0.8 #1.0
        cond_init_4[3,0,0,0] = 1.0#0.8 #1.0
        
        cond_init_4[1,4,0,0] = 1.0#0.8 #1.0
        cond_init_4[4,1,0,0] = 1.0#0.8 #1.0
        
        cond_init_4[2,5,0,0] = 1.0#0.8 #1.0
        cond_init_4[5,2,0,0] = 1.0#0.8 #1.0

        cond_init_4[3,6,0,0] = 1.0#0.8 #1.0
        cond_init_4[6,3,0,0] = 1.0#0.8 #1.0

        cond_init_4[4,7,0,0] = 1.0#0.8 #1.0
        cond_init_4[7,4,0,0] = 1.0#0.8 #1.0

        cond_init_4[5,8,0,0] = 1.0#0.8 #1.0
        cond_init_4[8,5,0,0] = 1.0#0.8 #1.0

        ## External edges
        cond_init_4[2,0,1,0]  = 1.0#1.0 #1.0
        cond_init_4[0,2,-1,0] = 1.0#1.0 #1.0

        cond_init_4[5,3,1,0]  = 1.0#1.0 #1.0
        cond_init_4[3,5,-1,0] = 1.0#1.0 #1.0

        cond_init_4[8,6,1,0]  = 1.0#1.0 #1.0
        cond_init_4[6,8,-1,0] = 1.0#1.0 #1.0

        cond_init_4[0,6,0,1]  = 1.0#1.0 #1.0
        cond_init_4[6,0,0,-1] = 1.0#1.0 #1.0

        cond_init_4[1,7,0,1]  = 1.0#1.0 #1.0
        cond_init_4[7,1,0,-1] = 1.0#1.0 #1.0

        cond_init_4[2,8,0,1]  = 1.0#1.0 #1.0
        cond_init_4[8,2,0,-1] = 1.0#1.0 #1.0

    return cond_init_4