#!/bin/bash
# To run, put in the terminal: bash micro_params.sh
# Define values for alpha and beta
betas=($(seq 0 0.01 0.1))
alphas=($(seq 0 0.1 1))

# Count the total number of iterations
total_iterations=$(echo "$alphas" | wc -w)

# Initialize the progress bar
progress_bar() {
    local current_iteration=$1
    local total_iterations=$((${#alphas[@]} * ${#betas[@]}))
    local progress=$((current_iteration * 100 / total_iterations))
    local bar_length=$((progress / 2))
    local bar=$(printf "%-${bar_length}s" "=" | tr ' ' '=')
    printf "\r[%-${bar_length}s] %d%%" "$bar" "$progress"
}

# Loop through all combinations of alpha and beta
iteration=0
start_time=$(date +%s)
for alpha in "${alphas[@]}"; do
    for beta in "${betas[@]}"; do
        iteration=$((iteration + 1))
        progress_bar "$iteration" "$total_iterations"

        echo " Running micro_main.py and macro_main.py with alpha=$alpha and beta=$beta"
        python3 multiscale/micro_main.py --alpha "$alpha" --beta "$beta"
        python3 multiscale/macro_main.py --alpha "$alpha" --beta "$beta"
    done
done
# Calculate and display the total execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: $execution_time seconds"
echo "Done!"