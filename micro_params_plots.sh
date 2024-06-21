#!/bin/bash
# To run, put in the terminal: bash micro_params.sh

# Define values for alpha and beta
alphas=($(seq 0 0.1 1))
betas=(0.01)
name="alpha_varying"
echo "Producing plots for the microscale model"
python3  multiscale/micro_plots.py --alphas "${alphas[@]}" --betas "${betas[@]}" --name "$name"
echo "Done!"

alphas2=(1.0)
betas2=($(seq 0 0.01 0.1))
name2="beta_varying"
echo "Producing plots for the microscale model"
python3  multiscale/micro_plots.py --alphas "${alphas2[@]}" --betas "${betas2[@]}" --name "$name2"
echo "Done!"