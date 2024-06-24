import argparse
import glob
import os
import json

def combine_results(directory, pattern, output_filename):

    combined_data = {}
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                # Extract the alpha and beta values from the filename
                filename = os.path.basename(filepath)
                alpha_beta_key = filename.replace(pattern.split('*')[0], '').replace('.json', '')
                alpha, beta = alpha_beta_key.split('_beta_')
                combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    full_output_path = os.path.join(directory, output_filename)
    try:
        with open(full_output_path, 'w') as file:
            json.dump(combined_data, file, indent=4)
    except Exception as e:
        print(f"Error writing to {full_output_path}: {e}")
    
    # Consider adding a confirmation or logging before removing files
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine micro or macro results.')
    parser.add_argument('type', choices=['micro', 'macro'], help='Specify whether to combine micro or macro results.')
    args = parser.parse_args()

    if args.type == 'micro':
        combine_results('multiscale/micro_results', 'micro_results_alpha_*.json', 'micro_results.json')
    elif args.type == 'macro':
        combine_results('multiscale/macro_results', 'macro_results_alpha_*.json', 'macro_results.json')



