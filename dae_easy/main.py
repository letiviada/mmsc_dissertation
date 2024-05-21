import numpy as np
import matplotlib.pyplot as plt
from solvers import solver_casadi, solver_scipy

def plot_solution(t_vals, x_vals, z_vals, method):
    plt.plot(t_vals, x_vals, label='x(t)')
    plt.plot(t_vals, z_vals, label='z(t)')
    #sol = lambda t: np.exp(2*t)
    #sol = lambda t: 1/3*(t**3+3)
    #plt.plot(t_vals,sol(t_vals),'--',label='exact')
    plt.xlabel('Time t')
    plt.ylabel('Value')
    plt.title(f'Solution of DAE using {method}')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
     # Initial condition
    initial_x = 0.0
    t_span = (0, 10)  # From 0 to 10
    # User chooses the method
    method = input("Choose the method (Scipy or CasADi): ").strip()
    
    if method == 'Scipy':
         t_vals, x_vals, z_vals = solver_scipy(initial_x, t_span)
    elif method == 'CasADi':
        t_vals, x_vals,z_vals = solver_casadi(initial_x,t_span)
    else:
        print("Invalid option. Please choose 'Scipy' or 'CasADi'.")
        return
    
    plot_solution(t_vals, x_vals, z_vals, method)


if __name__ == "__main__":
    main()