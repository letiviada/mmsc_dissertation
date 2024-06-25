#!/bin/bash
# To run, put in the terminal: bash micro_params.sh
# Define values for alpha and beta
#
alphas=$(seq 0 0.1 1)
betas=$(seq 0 0.1 0.1)
#alphas=(0.6)
betas=(0.1)
phis=(0.1)
# Start timing the entire script
script_start_time=$(date +%s)

# Start timing the micro computation
micro_computation_start_time=$(date +%s)
#python3 multiscale/micro_main.py --alphas $alphas --betas $betas
micro_computation_end_time=$(date +%s)
micro_computation_time=$((micro_computation_end_time - micro_computation_start_time))
echo "Micro computation time: $micro_computation_time seconds"

# Combine individual micro result files into one JSON file and delete them
micro_combine_start_time=$(date +%s)
#python3 multiscale/combine_results.py micro
micro_combine_end_time=$(date +%s)
micro_combine_time=$((micro_combine_end_time - micro_combine_start_time))
echo "Combining micro_results time: $micro_combine_time seconds"

# Start timing the macro computation
macro_computation_start_time=$(date +%s)
python3 multiscale/macro_main.py --alphas $alphas --betas $betas --phis $phis
macro_computation_end_time=$(date +%s)
macro_computation_time=$((macro_computation_end_time - macro_computation_start_time))
echo "Macro computation time: $macro_computation_time seconds"

# Combine individual micro result files into one JSON file and delete them
combine_start_time=$(date +%s)
python3 multiscale/combine_results.py macro_phi
combine_end_time=$(date +%s)
combine_time=$((combine_end_time - combine_start_time))
echo "Combining results time: $combine_time seconds"

# End timing the entire script
script_end_time=$(date +%s)
total_time=$((script_end_time - script_start_time))
echo "Total execution time: $total_time seconds"

echo "Done!"


