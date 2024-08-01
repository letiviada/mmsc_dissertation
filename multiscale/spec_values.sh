#!/bin/bash
# To run, put in the terminal: bash params_plots.sh
# Generate sequences
alpha_values=$(seq 0.1 0.1 1.0)
beta_values=$(seq 0.01 0.01 0.1)
phi=4.0

echo "Generating plots for the macroscale model"
python3 multiscale/macroscale/macro_plots.py --alpha_values $alpha_values --beta_values $beta_values --phi $phi #--parameter "{$parameter[@]}"
echo "Done!"

#echo "Generating plots for the performance indicators"
#python3 multiscale/performance_plots.py --alpha_values $alpha_values --beta_values $beta_values --phi $phi #--parameter "{$parameter[@]}"
#echo "Done!"