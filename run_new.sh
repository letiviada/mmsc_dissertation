#!/bin/bash
# To run, put in the terminal: bash micro_params.sh
# Define values for alpha and beta
betas=$(seq 0 0.01 0.1)
alphas=$(seq 0 0.1 1)

#computation_start_time=$(date +%s)
# Run the Python script with the list of alphas and betas
#python3 multiscale/micro_main1.py --alphas $alphas --betas $betas
#computation_end_time=$(date +%s)
# Calculate and display the computation time
#computation_time=$((computation_end_time - computation_start_time))
#echo "Computation time: $computation_time seconds"
# Combine individual result files into one JSON file
#python3 multiscale/combine_results.py

#echo "Done!"
#

# Start timing the entire script
script_start_time=$(date +%s)

# Start timing the micro computation
#micro_computation_start_time=$(date +%s)
#python3 multiscale/micro_main.py --alphas $alphas --betas $betas
#micro_computation_end_time=$(date +%s)
#micro_computation_time=$((micro_computation_end_time - micro_computation_start_time))
#echo "Micro computation time: $micro_computation_time seconds"

# Start timing the macro computation
#macro_computation_start_time=$(date +%s)
#python3 multiscale/macro_main.py --alphas $alphas --betas $betas
#macro_computation_end_time=$(date +%s)
#macro_computation_time=$((macro_computation_end_time - macro_computation_start_time))
#echo "Macro computation time: $macro_computation_time seconds"

# Combine individual micro result files into one JSON file and delete them
micro_combine_start_time=$(date +%s)
python3 multiscale/combine_results.py
micro_combine_end_time=$(date +%s)
micro_combine_time=$((micro_combine_end_time - micro_combine_start_time))
echo "Combining results time: $micro_combine_time seconds"


# End timing the entire script
script_end_time=$(date +%s)
total_time=$((script_end_time - script_start_time))
echo "Total execution time: $total_time seconds"

echo "Done!"

