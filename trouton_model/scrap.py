import numpy as np
import matplotlib.pyplot as plt

sol_u_initial = lambda x: 9*x+1
sol_h_initial = lambda x: np.ones(x.shape)
sol_u_final = lambda x: np.exp(np.log(10)*x)
sol_h_final = lambda x: 1/(np.exp(np.log(10)*x))

x_eval = np.linspace(0,1,100)
plt.figure()
plt.plot(x_eval, sol_u_final(x_eval),label=f'Initial u')
plt.tight_layout()
plt.show()


