import numpy as np
from scipy import linalg
def solve_W(G,l):
    """
    Solves for W given G.

    Parameters:
    G (np.ndarray): Input array of shape (4, 4, 3, 3).
    l (int): Length of the filtre

    Returns:
    np.ndarray: Output array of shape (4,).
    """
    N = 4
    R = 3
    #r_arr3 = np.array([-l,0,l])
    r_arr3 = l * np.array([0,1,-1])
    # Checked against Arkady's and seems fine
    Gk = np.sum(G, axis=(3,2,1))
    Gk_kronecker = np.eye(N) * Gk
    G_summed = np.sum(G, axis = (3,2))
    LHS = G_summed - Gk_kronecker

    RHS_4 = np.empty(shape = (N,N,R,R))
    for r in range(R):
        for s in range(R):
            RHS_4[:,:,r,s] = G[:,:,r,s] * r_arr3[r]
    RHS_2 = np.sum(a = np.sum(a=RHS_4,axis = 3),axis = 2)
    RHS_1 = -np.sum(a = RHS_2,axis = 1)
    W = linalg.solve(LHS,RHS_1)
    W = l * np.array([-0.25,  0.25, -0.25,  0.25])
    print(W)
    return W

def find_delta(W,l):
    """
    Finds delta given W.

    Parameters:
    W (np.ndarray): Input array of shape (N,).

    Returns:
    np.ndarray: Output array of shape (N, N, R), 
    component of the pressure difference in edge ijr per unit pressure gradient
    """
    N,R = 4,3
    W_matrix_2 = W[:, np.newaxis] - W
   #rl_array = np.array([-1, 0, 1])*l
    rl_array = np.array([0, 1,-1])*l
    # Initialize the tensor to store results (i, j, r)
    delta = np.zeros((N,N,R))
# Compute the values according to the formula for each r
    for r in range(R):
        delta[:,:,r] = W_matrix_2[:,:] -rl_array[r]
    return delta

def find_k(G,delta,l):
    """
    Finds k given G and delta

    Parameters:
    G (np.ndarray): Pore conductance (N,N,R,R).
    Delta (np.ndarray): Pressure difference (N,N,R)

    Returns:
    k (float): Scalar value k.
    """
    # Multiply delta by r
    N, R = 4,3
    delta_r = np.zeros((N,N,R))
    r_arr = np.array([0,1,-1])
    #r_arr = np.array([-1,0,1])
    for idx_r in range(R):
        delta_r[:,:,idx_r] = delta[:,:,idx_r] * r_arr[idx_r]
    # Make delta match the shape of G
    delta_4 = np.repeat(a=delta_r[:,:,:,np.newaxis],repeats=3,axis=-1)
    integrand = - G * delta_4
    k = (1 /(2 * 1)) * np.sum(integrand)
    return k

def find_j(alpha,G,delta,l):
    """
    Finds j given alpha, G, delta.

    Parameters:
    alpha (float): Stickiness
    G (np.ndarray): Pore conductance shape (N,N,R,R)
    Delta (np.ndarray): Pressure difference shape (N,N,R)
    l (float): Length of the filtre

    Returns:
    j (float): Scalar value j.
    """    
    # Make delta match the shape of G
    delta_4 = np.repeat(a=delta[:,:,:,np.newaxis],repeats=3,axis=-1)
    integrand =  - alpha * G * delta_4 * (1-np.heaviside(-G*delta_4, 1))
    j = -(1 / l) * np.sum(integrand)
    return j

def solve_G(alpha, beta, delta, G_previous, tau, dtau):
    """
    Solves for G given W and previous G.

    Parameters:
    alpha (float): Adhesitivity
    beta (float): Particle Size
    delta (np.ndarray): Pressure difference size (N,N,R)
    G_previous (np.ndarray): Previous G array of shape (N,N,R,R).
    tau (float): Current value of tau.

    Returns:
    G (np.ndarray): Output array of shape (N,N,R,R).
    """
    delta_4 = np.repeat(a=delta[:,:,:,np.newaxis],repeats=3,axis=-1)

    G = G_previous + dtau * (-alpha * beta * (G_previous ** (3/2)) * np.abs(delta_4))
    return G

def initial_G(initial_G_dict):
    """
    Solves for initial G given a dictionary containing non-zero values

    Parameters:
    initial_G_dict (dict): dictionary containing (i,j,r,s): value

    Returns:
    G (np.ndarray): Solution fot initial G of shape (4,4,3,3)
    """
    N,R = 4,3
    G = np.zeros((N,N,R,R))
    positions = initial_G_dict
    # Assign non-zero values to the tensor
    position= np.array(list(positions.keys()))
    values = np.array(list(positions.values()))
    idx_i = position[:, 0]
    idx_j = position[:, 1]
    idx_r = position[:, 2]
    idx_s = position[:, 3]
    G[idx_i, idx_j, idx_r, idx_s] = values
    print(G)
    return G