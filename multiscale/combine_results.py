import argparse
import glob
import os
import json
import zipfile
import re

def combine_results(directory, pattern,scale):

    combined_data = {}
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                filename = os.path.basename(filepath)
                if scale == 'macro_phi':
                    match = re.search(r'macro_results_alpha_(?P<alpha>\d*\.?\d+)_beta_(?P<beta>\d*\.?\d+)_phi_(?P<phi>\d*\.?\d+).json', filename)
                    if match:
                        alpha = match.group('alpha')
                        beta = match.group('beta')
                        phi = match.group('phi')
                    combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
                    output_filename = f'macro_results_phi_{phi}.json'
                elif scale == 'macro':
                    match = re.search(r'macro_results_alpha_(?P<alpha>\d*\.?\d+)_beta_(?P<beta>\d*\.?\d+)_phi_(?P<phi>\d*\.?\d+).json', filename)
                    if match:
                        alpha = match.group('alpha')
                        beta = match.group('beta')
                        phi = match.group('phi')
                    combined_data[f'(alpha,beta,phi)=({alpha},{beta},{phi})'] = data
                    output_filename = 'macro_results.json'
                
                elif scale == 'micro':
                    match = re.search(r'micro_results_alpha_(?P<alpha>\d*\.?\d+)_beta_(?P<beta>\d*\.?\d+).json', filename)
                    if match:
                        alpha = match.group('alpha')
                        beta = match.group('beta')
                    combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
                    output_filename = 'micro_results.json'
                elif scale =='performance_indicators':
                    match = re.search(r'performance_indicators_alpha_(?P<alpha>\d*\.?\d+)_beta_(?P<beta>\d*\.?\d+)_phi_(?P<phi>\d*\.?\d+).json', filename)
                    if match:
                        alpha = match.group('alpha')
                        beta = match.group('beta')
                        phi = match.group('phi')
                    combined_data[f'(alpha,beta)=({alpha},{beta})'] = data
                    output_filename = f'performance_indicators_phi_{phi}.json'
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
    

    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine micro or macro results.')
    parser.add_argument('type', choices=['micro', 'macro','macro_phi','performance_indicators'], help='Specify whether to combine micro or macro results.')
    parser.add_argument('--num_runs', type = int, help = 'Number of runs we simulated')
    args = parser.parse_args()
    
    num_run = args.num_runs

    if num_run > 1:
        if args.type == 'micro':
            combine_results('multiscale/results/poly-dispersed/microscale', 'micro_results_alpha_*.json',scale = 'micro')
        elif args.type == 'macro_phi':
            combine_results(directory='multiscale/results/poly-dispersed/macroscale', pattern='macro_results_alpha_*.json',scale='macro_phi')
        elif args.type == 'performance_indicators':
            combine_results(directory = 'multiscale/results/poly-dispersed/performance_indicators', pattern = 'performance_indicators_alpha_*.json',scale='performance_indicators' )
    else:
        if args.type == 'micro':
            combine_results('multiscale/results/mono-dispersed/microscale', 'micro_results_alpha_*.json',scale = 'micro')
        elif args.type == 'macro_phi':
            combine_results(directory='multiscale/results/mono-dispersed/macroscale', pattern='macro_results_alpha_*.json',scale = 'macro_phi')
        elif args.type == 'performance_indicators':
            combine_results(directory = 'multiscale/results/mono-dispersed/performance_indicators', pattern = 'performance_indicators_alpha_*.json',scale='performance_indicators' )




        #combine_results('multiscale/results/microscale/full_output', 'micro_results_alpha_*.json',scale = 'micro')
        #zip_filename = 'multiscale/results/microscale/full_output/micro_results.zip'
        #with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
          #  micro_results_path = 'multiscale/results/microscale/full_output/micro_results.json'
          #  zipf.write(micro_results_path, os.path.basename(micro_results_path))
          #  print(f"Compressed {micro_results_path} into {zip_filename}")
   # elif args.type == 'macro':
        #combine_results('multiscale/results/macroscale', 'macro_results_alpha_*.json', 'macro_results.json','macro')