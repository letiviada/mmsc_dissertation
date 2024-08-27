#!/bin/bash
# To run, put in the terminal: bash params_plots.sh
phi=2.0

# Define values for alpha and beta
alphas=$(seq 0.2 0.2 1.0)
betas=(0.06)
name="beta_0.06_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.06"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.6)
betas2=$(seq 0.02 0.02 0.1)
name2="alpha_0.6_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"



