import matplotlib.pyplot as plt
from solver import solver_casadi

def plot_solution(t_vals, x_vals, z_vals):
    [plt.plot(t_vals, x_vals[i], label=f'x{i}(x)') for i in range(len(x_vals))]
    plt.plot(t_vals, z_vals, label='z(t)')
    plt.xlabel('Time t')
    plt.ylabel('Value')
    plt.title(f'Solution of DAE using CasADi')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
     # Initial condition
    initial_x = [0,1,0.5]
    t_span = (0, 3)  # From 0 to 10
    t_vals, x_vals,z_vals = solver_casadi(initial_x,t_span,len(initial_x),initial_guess=0)
    plot_solution(t_vals, x_vals, z_vals)


if __name__ == "__main__":
    main()