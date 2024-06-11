from casadi import *
from solver import solver_dae, run, reshape

def main():
    # Retrieve data from the microscale system
    k = np.ones(101)
    j = 2*np.ones(101)
    tau_eval = np.linspace(-10,10,101)

    # Spatial domain
    nx = 5
    x_eval = np.linspace(0,1,nx)
    # Time domain
    nt=11
    t_eval = np.linspace(0,10,nt)

    # Initial Conditions
    c00 = 1.0
    c0 = np.zeros((nx-1,1))
    tau0 = np.zeros((nx,1))
    x0 = vertcat(c00,c0,tau0)
    z0 = np.zeros((nx+1,1))

    # Define solver
    F = solver_dae(k,j,tau_eval,x_eval,t_eval)
    # Run code
    x_res, z_res = run(F,x0,z0)

    # Reshape
    c,tau,u,psi = reshape(x_res,z_res,nt,nx)
    print(f'C is: {c}')
    print(f'tau is: {tau}')
    print(f'u is: {u}')
    print(f'psi is: {psi}')
if __name__ == '__main__':
    main()