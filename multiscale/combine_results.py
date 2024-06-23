import glob
import os
import json

def combine_micro_results(directory='multiscale/', output_filename='multiscale/micro_results.json'):
    combined_data = {}
    for filepath in glob.glob(os.path.join(directory, 'results_alpha_*.json')):
        with open(filepath, 'r') as file:
            data = json.load(file)
            # Extract the alpha and beta values from the filename
            filename = os.path.basename(filepath)
            alpha_beta_key = filename.replace('results_alpha_', '').replace('.json', '')
            alpha, beta = alpha_beta_key.split('_beta_')
            combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
    
    with open(output_filename, 'w') as file:
        json.dump(combined_data, file, indent=4)
    
    # Remove individual result files
    for filepath in glob.glob(os.path.join(directory, 'results_alpha_*.json')):
        os.remove(filepath)


def combine_macro_results(directory='multiscale/', output_filename='multiscale/macro_results.json'):
    combined_data = {}
    for filepath in glob.glob(os.path.join(directory, 'macro_results_alpha_*.json')):
        with open(filepath, 'r') as file:
            data = json.load(file)
            # Extract the alpha and beta values from the filename
            filename = os.path.basename(filepath)
            alpha_beta_key = filename.replace('macro_results_alpha_', '').replace('.json', '')
            alpha, beta = alpha_beta_key.split('_beta_')
            combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
    
    with open(output_filename, 'w') as file:
        json.dump(combined_data, file, indent=4)
    
    # Remove individual result files
    for filepath in glob.glob(os.path.join(directory, 'macro_results_alpha_*.json')):
        os.remove(filepath)


if __name__ == "__main__":
    combine_micro_results()
    combine_macro_results()

