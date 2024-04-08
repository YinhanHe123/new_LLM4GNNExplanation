import pickle

import pickle



def load_pickle_file(file_path):
    """
    Loads and returns the content of a pickle file.

    Parameters:
    - file_path (str): The path to the .pkl file to be loaded.

    Returns:
    - The content of the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def write_to_text_file(text_file_path, data):
    """
    Writes the given data to a text file.

    Parameters:
    - text_file_path (str): The path to the .txt file to be written.
    - data: Data to be written to the file. Assumes data can be converted to a string.
    """
    try:
        with open(text_file_path, 'w') as file:
            # Convert data to string and write it
            for i in range(len(data)):
                file.write(str(data[i])+'\n')
            # file.write(str(data)+'\n')
        print("Data written successfully to text file.")
    except Exception as e:
        print(f"Error writing to text file: {e}")

if __name__ == "__main__":
    # pickle_file_path_list_mutag = ['/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/Mutagenicity/CF_GNNExplainer/stock_cf_gnn_explainer_mutag_exp-3_smiles.pkl',
    #                             '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/Mutagenicity/CLEAR/stock_clear_mutag_exp-3_smiles.pkl',
    #                             '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/Mutagenicity/GNNExplainer/stock_gnn_explainer_mutag_exp-3_smiles.pkl',
    #                             '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/Mutagenicity/RegExplainer/stock_reg_explainer_mutag_exp-3_smiles.pkl']
    
    # pickle_file_path_list_bbbp = ['/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/BBBP/CF_GNNExplainer/CF_GNNExplainer_BBBP_exp-3_smiles.pkl',
    #                             '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/BBBP/CLEAR/CLEAR_BBBP_exp-3_smiles.pkl',
    #                             '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/BBBP/GNNExplainer/GNNExplainer_BBBP_exp-3_smiles.pkl',
    #                             '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/BBBP/RegExplainer/RegExplainer_BBBP_exp-3_smiles.pkl']
    pickle_file_path_list_bbbp = ['/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/BBBP/GNN_Explainer/GNN_Explainer_BBBP_exp-3_smiles.pkl']


    for pickle_file_path in pickle_file_path_list_bbbp:
        # pickle_file_path = '/home/nee7ne/research_code/new_LLM4GNNExplanation/plots/llm-gce-baseline-outputs/Mutagenicity/CF_GNNExplainer/stock_cf_gnn_explainer_mutag_exp-3_smiles.pkl'
        data = load_pickle_file(pickle_file_path)
        if data is not None:
            print("Data loaded successfully!")

            # Generate the text file path by replacing .pkl with .txt
            csv_file_path = pickle_file_path.rsplit('.', 1)[0] + '.csv'
            
            # Write the data to the text file
            write_to_text_file(csv_file_path, data)
        else:
            print("Failed to load data.")
