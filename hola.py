import matplotlib.pyplot as plt
import numpy as np
N,R = 4,3
G =  np.random.rand(144).reshape((4,4,3,3))

lhs1 = np.empty(shape = (N,N,R,R))
for r in range(R):
    for s in range(R):
        lhs1[:,:,r,s] = G[:,:,r,s] - np.diag(np.sum(a = G[:,:,r,s],axis = 1))
lhs =  np.sum(a=np.sum(a=lhs1, axis=3), axis=2)

Gk = np.sum(G, axis=(3,2,1))
Gk_kronecker = np.eye(4) * Gk
Gk_kronecker = np.diag(Gk)
G_summed = np.sum(G, axis = (3,2))
LHS = G_summed - Gk_kronecker
print(lhs - LHS)
print(lhs)
print(LHS)
