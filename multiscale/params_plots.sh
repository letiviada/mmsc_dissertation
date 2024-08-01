#!/bin/bash
# To run, put in the terminal: bash params_plots.sh
phi=4.0

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.01)
name="beta_0.01_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.01"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.1)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.1_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.02)
name="beta_0.02_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.02"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.2)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.2_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.03)
name="beta_0.03_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.03"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.3)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.3_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.4)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.4_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.04)
name="beta_0.04_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.04"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.5)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.5_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.05)
name="beta_0.05_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.05"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"


# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.06)
name="beta_0.06_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.06"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.6)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.6_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.07)
name="beta_0.07_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.07"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.7)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.7_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"


# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.08)
name="beta_0.08_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.08"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.8)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.8_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.09)
name="beta_0.09_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.09"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(0.9)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_0.9_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"

# Define values for alpha and beta
alphas=$(seq 0.1 0.1 1.0)
betas=(0.1)
name="beta_0.1_alpha_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with alpha varying and beta = 0.1"
python3  multiscale/sweep_params_plots.py --alphas $alphas --betas "${betas[@]}" --name "$name" --phis "${phi[@]}"
echo "Done!"

alphas2=(1.0)
betas2=$(seq 0.01 0.01 0.1)
name2="alpha_1.0_beta_varying"
echo "Producing plots for k(s),j(s),u(t),c(t,x) with beta varying"
python3  multiscale/sweep_params_plots.py --alphas "${alphas2[@]}" --betas $betas2 --name "$name2" --phis "${phi[@]}"
echo "Done!"


