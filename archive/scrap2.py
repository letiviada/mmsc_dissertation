import numpy as np

def delta_loop1(W, l=1):
    """
    Function that creates the (tau,N,N,3) tensor Delta_{ij}^{r}
    """
    tau, N= W.shape
    R = np.array([-1, 0, 1])
    Delta = np.zeros((tau, N, N, len(R)))
    for batch in range(tau):
        for i in range(N):
            for j in range(N):
                for k, r in enumerate(R):
                    Delta[batch, i, j, k] = W[batch, i] - (W[batch, j] + r * l)
    return Delta

def delta_broadcast1(W):
   R = np.array([-1, 0, 1])
   W_i = W[:, :, np.newaxis, np.newaxis]  # Shape (s, N, 1, 1)
   W_j = W[:, np.newaxis, :, np.newaxis]  # Shape (s, 1, N, 1)
   r_l = R[np.newaxis, np.newaxis, np.newaxis, :]  # Shape (1, 1, 1, 3)
   Delta_broadcast = W_i - (W_j + r_l)  # Shape (s, N, N, 3)
   return Delta_broadcast
# ------------------------------------------------------
def delta_loop(W, l=1):
    N = len(W)
    R = np.array([-1, 0, 1])
    Delta = np.zeros((N, N, len(R)))
    for i in range(N):
        for j in range(N):
            for k, r in enumerate(R):
                Delta[i, j, k] = W[i] - (W[j] + r * l)
    return Delta
