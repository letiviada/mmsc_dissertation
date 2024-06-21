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
    delta = np.empty(shape=(N,N,R)) # delta[i,j,r]

    W_matrix_2 = W[:, np.newaxis] - W # W_i - W_j
    rl_array = np.array([0, 1,-1])*l # r*l
    for r in range(R):
        delta[:,:,r] = W_matrix_2[:,:] -rl_array[r] # W_i - W_j - r*l
    return delta

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
    return k

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
    j = - (1/l)*np.sum(integrand)
    return j

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