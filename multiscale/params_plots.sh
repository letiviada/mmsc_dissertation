#!/bin/bash
# To run, put in the terminal: bash params_plots.sh
phi=(4.0)

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 0.9)
betas=(0.02)
name="beta_0.02_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.02"
#python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.2)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.2_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
#python3  multiscale/micro_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Generate sequences
alpha_values=$(seq 0.3 0.1 1.0)
beta_values=$(seq 0.03 0.01 0.1)

#parameter=("auxiliar_variable") # "concentration" #"darcy_velocity", "permeability", "adhesivity", "reactivity")
echo "Generating plots for the macroscale model"
#python3 multiscale/macro_plots.py --alpha_values $alpha_values --beta_values $beta_values --phi $phi #--parameter "{$parameter[@]}"
echo "Done!"

# 
echo "Generating plots for the performance indicators"
#python3 multiscale/performance_plots.py --alpha_values $alpha_values --beta_values $beta_values --phi $phi #--parameter "{$parameter[@]}"
echo "Done!"