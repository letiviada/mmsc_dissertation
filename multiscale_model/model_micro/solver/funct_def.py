import numpy as np

def solve_W(G,l=1):
    """
    Solves for W given G.

    Parameters:
    G (np.ndarray): Input array of shape (4, 4, 3, 3).
    l (int): Length of the filtre

    Returns:
    np.ndarray: Output array of shape (4,).
    """
    def get_r():
        shape = (4,4,3,3)
        r_tensor = np.zeros(shape)

            # Assign values based on r and s
        for r in range(-1, 2):
            for s in range(-1, 2):
                if (r, s) in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    value = 0
                else:
                    value = r + s
                r_tensor[:, :, r + 1, s + 1] = value
                
        return r_tensor
    r = get_r()
    Gk = np.sum(G, axis=(3,2,1))
    Gk_kronecker = np.eye(4) * Gk
    G_summed = np.sum(G, axis = (3,2))
    LHS = G_summed - Gk_kronecker
    integrand = G * r * l
    RHS = np.sum(integrand, axis = (3,2,1))
    #W = np.linalg.solve(LHS,RHS)
    W = np.array([-1,1,1,-1])
    return W

def find_delta(W,l=1):
    """
    Finds delta given W.

    Parameters:
    W (np.ndarray): Input array of shape (4,).

    Returns:
    np.ndarray: Output array of shape (4, 4, 3), 
    component of the pressure difference in edge ijr per unit pressure gradient
    """
    diff_matrix = W[:, np.newaxis] - W
    R = np.array([-1, 0, 1])*l
    # Initialize the tensor to store results (i, j, r)
    delta = np.zeros((4, 4, 3))
# Compute the values according to the formula for each r
    for idx, r_l in enumerate(R):
        delta[:, :, idx] = diff_matrix - (r_l)
    return delta

def find_k(G,delta,l=1):
    """
    Finds k given G and delta

    Parameters:
    G (np.ndarray): Pore conductance (4,4,3,3).
    Delta (np.ndarray): Pressure difference (4,4,3)

    Returns:
    k (float): Scalar value k.
    """
    # Multiply delta by r
    delta_r = np.zeros((4,4,3))
    r = np.array([-1,0,1])
    for idx_r in range(len(r)):
        delta_r[:,:,idx_r] = delta[:,:,idx_r] * r[idx_r]
    # Make delta match the shape of G
    result_exp = np.expand_dims(delta_r, axis=-1)
    delta_ext = np.tile(result_exp, (1, 1, 1, 3))
    integrand = - G * delta_ext
    k = (1 / (2*l)) * np.sum(integrand)
    return k

def find_j(alpha,G,delta,k, l=1):
    """
    Finds j given alpha, G, delta.

    Parameters:
    alpha (float): Stickiness
    G (np.ndarray): Pore conductance shape (4,4,3,3)
    Delta (np.ndarray): Pressure difference shape (4,4,3)
    l (float): Length of the filtre

    Returns:
    j (float): Scalar value j.
    """    
    # Make delta match the shape of G
    result_exp = np.expand_dims(delta, axis=-1)
    delta_ext = np.tile(result_exp, (1, 1, 1, 3))
    integrand =  alpha * G * delta_ext * (1-np.heaviside(G*delta_ext, 0))
    j = -(1 / l) * np.sum(integrand, axis = (3,2,1,0))
    return j

def solve_G(alpha, beta, delta, G_previous, tau, dtau):
    """
    Solves for G given W and previous G.

    Parameters:
    alpha (float): Adhesitivity
    beta (float): Particle Size
    delta (np.ndarray): Pressure difference size (4,4,3)
    G_previous (np.ndarray): Previous G array of shape (4, 4, 3, 3).
    tau (float): Current value of tau.

    Returns:
    G (np.ndarray): Output array of shape (4, 4, 3, 3).
    """
    result_exp = np.expand_dims(delta, axis=-1)
    delta_ext = np.tile(result_exp, (1, 1, 1, 3))

    G = G_previous + dtau * (-alpha * beta * G_previous ** (3/2) * np.abs(delta_ext))
    return G

def initial_G(initial_G_dict):
    """
    Solves for initial G given a dictionary containing non-zero values

    Parameters:
    initial_G_dict (dict): dictionary containing (i,j,r,s): value

    Returns:
    G (np.ndarray): Solution fot initial G of shape (4,4,3,3)
    """
    G = np.zeros((4,4,3,3))
    positions = initial_G_dict
    # Assign non-zero values to the tensor
    position= np.array(list(positions.keys()))
    values = np.array(list(positions.values()))
    idx_i = position[:, 0]
    idx_j = position[:, 1]
    idx_r = position[:, 2]
    idx_s = position[:, 3]
    G[idx_i, idx_j, idx_r, idx_s] = values
    return G