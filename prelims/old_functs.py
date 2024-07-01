import json
import os
def save_macro_results(alpha, beta,phi, output_dict, directory='multiscale/results/macroscale'):
    """
    Saves the results to a JSON file, converting NumPy arrays to lists.

    Parameters:
    alpha (float): Adhesivity
    beta (float): Particle size
    output_dict (dict): Dictionary containing the output values of the macroscale model.
    directory (str): Directory to save the results to.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate the filename dynamically based on alpha and beta values
    filename = os.path.join(directory, f'macro_results_alpha_{alpha}_beta_{beta}_phi_{phi}.json')

    # Save as JSON
    with open(filename, 'w') as file:
        json.dump(output_dict, file, indent=4)