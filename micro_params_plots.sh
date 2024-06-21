#!/bin/bash
# To run, put in the terminal: bash micro_params.sh

# Define values for alpha and beta
alphas=($(seq 0 0.01 0.1))
betas=(1)
echo "Producing plots for the microscale model"
python3  multiscale/micro_plots.py --alpha "${alphas[@]}" --beta "$betas"