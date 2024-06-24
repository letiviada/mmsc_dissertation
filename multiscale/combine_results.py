import argparse
import glob
import os
import json
import re

def combine_results(directory, pattern, output_filename,scale):

    combined_data = {}
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                filename = os.path.basename(filepath)
                if scale == 'macro':
                    match = re.search(r'macro_results_alpha_(?P<alpha>\d*\.?\d+)_beta_(?P<beta>\d*\.?\d+)_phi_(?P<phi>\d*\.?\d+).json', filename)
                    if match:
                        alpha = match.group('alpha')
                        beta = match.group('beta')
                        phi = match.group('phi')
                    combined_data[f'(alpha,beta,phi)=({alpha},{beta},{phi})'] = data
                elif scale == 'micro':
                    match = re.search(r'micro_results_alpha_(?P<alpha>\d*\.?\d+)_beta_(?P<beta>\d*\.?\d+).json', filename)
                    if match:
                        alpha = match.group('alpha')
                        beta = match.group('beta')
                    combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    full_output_path = os.path.join(directory, output_filename)
    if os.path.exists(full_output_path):
        try:
            with open(full_output_path, 'r') as file:
                existing_data = json.load(file)
            existing_data.update(combined_data)
            combined_data = existing_data
        except Exception as e:
            print(f"Error reading existing {output_filename}: {e}")

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
        combine_results('multiscale/results/microscale', 'micro_results_alpha_*.json', 'micro_results.json','micro')
    elif args.type == 'macro':
        combine_results('multiscale/results/macroscale', 'macro_results_alpha_*.json', 'macro_results.json','macro')



