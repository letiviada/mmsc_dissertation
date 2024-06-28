import os
import json
import numpy as np

def check_json_files(filename,directory = 'multiscale/results/macroscale', file_extension = ".json"):
    alpha_values = np.around(np.linspace(0.2, 0.7, 6), 2)
    beta_values = np.around(np.linspace(0.01, 0.09, 9), 2)
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
#check_json_files("macro_results_phi_0.1.json")
#check_json_files("macro_results_phi_0.2.json")
#check_json_files("macro_results_phi_0.6.json")
check_json_files("macro_results_phi_0.7.json")
#check_json_files("macro_results_phi_0.8.json")
#check_json_files("macro_results_phi_0.9.json")
