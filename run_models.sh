#!/bin/bash
# To run, put in the terminal: bash micro_params.sh
# Define values for alpha and beta
#
alphas=$(seq 0.3 0.1 1.0)
betas=$(seq 0.03 0.01 0.1)
phis=(0.5)
# Start timing the entire script
script_start_time=$(date +%s)

# Start timing the micro computation
micro_computation_start_time=$(date +%s)
#python3 multiscale/micro_main.py --alphas $alphas --betas $betas
micro_computation_end_time=$(date +%s)
micro_computation_time=$((micro_computation_end_time - micro_computation_start_time))
minutes=$((micro_computation_time / 60))
seconds=$((micro_computation_time % 60))
echo "Micro computation time: $minutes minutes and $seconds seconds"

# Combine individual micro result files into one JSON file and delete them
micro_combine_start_time=$(date +%s)
#python3 multiscale/combine_results.py micro
micro_combine_end_time=$(date +%s)
micro_combine_time=$((micro_combine_end_time - micro_combine_start_time))
minutes=$((micro_combine_time / 60))
seconds=$((micro_combine_time % 60))
echo "Combining micro_results time: $minutes minutes and $seconds seconds"

# Start timing the macro computation
macro_computation_start_time=$(date +%s)
#python3 multiscale/macro_main.py --alphas $alphas --betas $betas --phis $phis
macro_computation_end_time=$(date +%s)
macro_computation_time=$((macro_computation_end_time - macro_computation_start_time))
minutes=$((macro_computation_time / 60))
seconds=$((macro_computation_time % 60))
echo "Macro computation time: $minutes minutes and $seconds seconds"

# Combine individual micro result files into one JSON file and delete them
combine_start_time=$(date +%s)
#python3 multiscale/combine_results.py macro_phi
combine_end_time=$(date +%s)
combine_time=$((combine_end_time - combine_start_time))
minutes=$((combine_time / 60))
seconds=$((combine_time % 60))
echo "Combining results time: $minutes minutes and $seconds seconds"


# Performance indicators computation
perf_indicators_start=$(date +%s)
python3 multiscale/perfo_indicators.py --alphas $alphas --betas $betas --phis $phis
perf_indicators_end=$(date +%s)
perf_indicators_time=$((perf_indicators_end - perf_indicators_start))
minutes=$((perf_indicators_end_time / 60))
seconds=$((perf_indicators_end_time % 60))
echo "Performance Indicators computation time: $minutes minutes and $seconds seconds"

# Combine individual micro result files into one JSON file and delete them
combine_start_time=$(date +%s)
python3 multiscale/combine_results.py performance_indicators
combine_end_time=$(date +%s)
combine_time=$((combine_end_time - combine_start_time))
minutes=$((combine_time / 60))
seconds=$((combine_time % 60))
echo "Combining results time: $minutes minutes and $seconds seconds"
# End timing the entire script
script_end_time=$(date +%s)
total_time=$((script_end_time - script_start_time))
min=$((total_time / 60))
sec=$((total_time % 60))
echo "Total execution time: $min minutes and $sec seconds"
echo "Done!"


