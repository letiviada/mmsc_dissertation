import os
import json
import numpy as np

def check_json_files(filename,directory = 'multiscale/results/mono-dispersed/', file_extension = ".json"):
    alpha_values = np.around(np.arange(0.3, 0.95, 0.05), 2)
    beta_values = np.around(np.arange(0.03, 0.1, 0.004), 3)
# Construct the full path to the file
    filepath = os.path.join(directory, filename)
    
    # Check if the file exists and matches the file extension
    if os.path.isfile(filepath) and filename.endswith(file_extension):
        with open(filepath) as file:
            data = json.load(file)
            for alpha in alpha_values:
                for beta in beta_values:
                    key = f"(alpha,beta)=({alpha},{beta})"
                    if key in data.keys():
                        continue
                        
                    else:
                        print(f"Key {key} not found in file {filename}")
    else:
        print(f"File '{filename}' not found in directory '{directory}' or does not match the file extension '{file_extension}'.")
    print('Done')
# Call the function  
check_json_files("performance_indicators/performance_indicators_phi_1.0.json")
check_json_files("macroscale/macro_results_phi_1.0.json")


