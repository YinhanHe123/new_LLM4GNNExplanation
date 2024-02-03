# import openai
# import csv
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from tqdm import tqdm
# import argparse

# def read_csv(file_path):
#     file = open(file_path, "r")
#     data = list(csv.reader(file, delimiter=","))
#     if(len(data[0]) == 1):
#         processed_data = [float(item[0]) for item in data]
#     else:
#         processed_data = [[float(item) for item in row] for row in data]
#     return processed_data

# def preprocess_data(dataset):
#     dataset_path = './' + dataset + "/"
#     if(dataset == "AIDS"):
#         edges = read_csv(dataset_path + "AIDS.edges")
#         graph_idx = read_csv(dataset_path + "AIDS.graph_idx")
#         graph_labels = read_csv(dataset_path + "AIDS.graph_labels")
#         link_labels = read_csv(dataset_path + "AIDS.link_labels")
#         # node_attrs = read_csv(dataset_path + "AIDS.node_attrs")
#         node_labels = read_csv(dataset_path + "AIDS.node_labels")
#         return edges, graph_idx, graph_labels, link_labels, node_labels
#     elif(dataset == "Mutagenicity"):
#         edges = read_csv(dataset_path + "Mutagenicity.edges")
#         graph_idx = read_csv(dataset_path + "Mutagenicity.graph_idx")
#         graph_labels = read_csv(dataset_path + "Mutagenicity.graph_labels")
#         link_labels = read_csv(dataset_path + "Mutagenicity.link_labels")
#         # node_attrs = read_csv(dataset_path + "Mutagenicity.node_attrs")
#         node_labels = read_csv(dataset_path + "Mutagenicity.node_labels")
#         return edges, graph_idx, graph_labels, link_labels, node_labels
#     return None
        

# def get_description(molecule_data):
#     response = openai.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"Describe this molecule that is given in smiles representation using one sentence of less than 50 words: {molecule_data}",
#             },
#         ],
#     )
#     return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens


# def parse_args():
#     parser = argparse.ArgumentParser(description='Annotate Graphs')
#     parser.add_argument('--dataset', type=str, default='AIDS', help='Dataset to use')
#     args = parser.parse_args()
#     return args



# if __name__ == "__main__":
#     args = parse_args()
#     if args.dataset == "AIDS":
#         NODE_LABEL_MAP = {0.0: 'C', 1.0: 'O', 2.0: 'N', 3.0: 'Cl', 4.0: 'F', 5.0: 'S', 
#                   6.0: 'Se', 7.0: 'P', 8.0: 'Na', 9.0: 'I', 10.0: 'Co', 11.0: 'Br', 
#                   12.0: 'Li', 13.0: 'Si', 14.0: 'Mg', 15.0: 'Cu', 16.0: 'As', 
#                   17.0: 'B', 18.0: 'Pt', 19.0: 'Ru', 20.0: 'K', 21.0: 'Pd', 
#                   22.0: 'Au', 23.0: 'Te', 24.0: 'W', 25.0: 'Rh', 26.0: 'Zn', 27.0: 'Bi', 
#                   28.0: 'Pb', 29.0: 'Ge', 30.0: 'Sb', 31.0: 'Sn', 32.0: 'Ga', 33.0: 'Hg', 
#                   34.0: 'Ho', 35.0: 'Tl', 36.0: 'Ni', 37.0: 'Tb'}
#     elif args.dataset == "Mutagenicity":
#         NODE_LABEL_MAP = {
#             0.0: 'C', 1.0: 'O', 2.0: 'Cl', 3.0: 'H', 4.0: 'N', 5.0: 'F', 6.0: 'Br', 
#             7.0: 'S', 8.0: 'P', 9.0: 'I', 10.0: 'Na', 11.0: 'K', 12.0: 'Li', 13.0: 'Ca'
#         }
#     edges, graph_idx, graph_labels, link_labels, node_labels = preprocess_data(args.dataset)
#     smiles_list = []
#     cur_idx = 0
#     for i in tqdm(range(len(graph_labels))):
#         node_list = {}
#         mol = Chem.RWMol()
#         while(cur_idx < len(graph_idx) and graph_idx[cur_idx] - 1 == i):
#             a = Chem.Atom(NODE_LABEL_MAP[node_labels[cur_idx]])
#             molIdx = mol.AddAtom(a)
#             node_list[cur_idx] =  molIdx
#             cur_idx += 1

#         edge_idxs = [idx for idx, x in enumerate(edges) if x[0] in node_list]
#         for edge_idx in edge_idxs:
#             x,y = edges[edge_idx][0] - 1, edges[edge_idx][1] - 1
#             if(y < x):
#                 continue
#             if link_labels[edge_idx] == 0:
#                 bond_type = Chem.rdchem.BondType.SINGLE
#                 mol.AddBond(node_list[x], node_list[y], bond_type)
#             elif link_labels[edge_idx] == 1:
#                 bond_type = Chem.rdchem.BondType.DOUBLE
#                 mol.AddBond(node_list[x], node_list[y], bond_type)
#             elif link_labels[edge_idx] == 2:
#                 bond_type = Chem.rdchem.BondType.TRIPLE
#                 mol.AddBond(node_list[x], node_list[y], bond_type)
#         smiles_list.append(Chem.MolToSmiles(mol.GetMol()))    
#     DATASET_ROOT_PATH = "./"+args.dataset+"/"
#     openai.api_key = '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'
#     args = parse_args()
#     save_path = "./"+args.dataset+"_output.csv"
#     output = open(save_path, 'w')
#     writer = csv.writer(output)
#     for smile in tqdm(smiles_list):
#         molecule = smile 
#         message, completion, prompt = get_description(molecule)
#         writer.writerow([molecule, message, completion, prompt])


# from openai import OpenAI

# client = OpenAI(api_key="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")

# stream = client.chat.completions.create(
#     model="gpt-3.5-turbo-1106",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": "What is the meaning of life?"
#         }
#     ]
# )



# for chunk in stream:
#     print(chunk)


# from rdkit import Chem
# print(Chem.MolFromSmiles("C#[C]1#C[Br][C]2=[C]3#[C]#[C]4=3=[C]3=[C]5(=[C](#C)[Cl]6[C]785[C](#[C]=1)=C=[N]1[Cl]5(#[S]=2#[C]26[N](=[Cl])[C](#[C]4#[Br]2[C]1#[F])#C)#S8[Cl]75)#[Ca]3.I[IH][I][I]I.CP.FI.[H][H]"))


from rdkit import Chem
from rdkit.Chem import rdchem
import rdkit.Chem.AllChem as AllChem

# Original SMILES string
smiles = "C#[C]1#C[Br][C]2=[C]3#[C]#[C]4=3=[C]3=[C]5(=[C](#C)[Cl]6[C]785[C](#[C]=1)=C=[N]1[Cl]5(#[S]=2#[C]26[N](=[Cl])[C](#[C]4#[Br]2[C]1#[F])#C)#S8[Cl]75)#[Ca]3.I[IH][I][I]I.CP.FI.[H][H]"

# Load the molecule from the SMILES string
mol = Chem.MolFromSmiles(smiles)

# If the molecule is successfully loaded, attempt to clean it up
if mol:
    # Attempt to sanitize the molecule, catching any exceptions if the molecule is invalid
    try:
        Chem.SanitizeMol(mol)
        
        # If the molecule is valid but needs minimal perturbation, manually adjust the structure
        # The specific adjustments would depend on the exact requirements and the structure itself
        # This step is where you would typically correct valence issues, remove or add atoms/bonds, etc.
        # However, without specific instructions for adjustments, we proceed to check and correct valences
        
        # Dummy step for illustration; in practice, this would involve specific corrections
        # Here, we simply generate the cleaned SMILES without changes
        cleaned_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        # If the molecule cannot be sanitized (invalid structure), then we need to manually adjust it
        # Placeholder for error handling; in a real scenario, we would correct the structure here
        cleaned_smiles = "Error: Cannot be sanitized without specific corrections."
else:
    cleaned_smiles = "Error: Invalid SMILES input."

print(cleaned_smiles)