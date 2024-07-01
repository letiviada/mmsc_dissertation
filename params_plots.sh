#!/bin/bash
# To run, put in the terminal: bash params_plots.sh

# Define values for alpha and beta
alphas=($(seq 0 0.1 1))
betas=(0.01)
name="alpha_varying"
echo "Producing plots for the microscale model"
#python3  multiscale/micro_plots.py --alphas "${alphas[@]}" --betas "${betas[@]}" --name "$name"
echo "Done!"

alphas2=(1.0)
betas2=($(seq 0 0.01 0.1))
name2="beta_varying"
echo "Producing plots for the microscale model"
#python3  multiscale/micro_plots.py --alphas "${alphas2[@]}" --betas "${betas2[@]}" --name "$name2"
echo "Done!"

# Generate sequences
alpha_values=$(seq 0.1 0.1 1.0)
beta_values=$(seq 0.01 0.01 0.1)
phi=(0.5)
alpha_values=(1.0)
beta_values=(0.09)
#parameter=("auxiliar_variable") # "concentration" #"darcy_velocity", "permeability", "adhesivity", "reactivity")
# Call the Python script with the sequences
echo "Generating plots for the macroscale model"
python3 multiscale/macro_plots.py --alpha_values $alpha_values --beta_values $beta_values --phi $phi #--parameter "{$parameter[@]}"
echo "Done!"
