import os
import sys

import pandas as pd
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(__file__).split("data")[0])
from utils.data_load import get_graphs_from_data, preprocess_graph_data
from utils.smiles import graph_to_smiles
from rdkit import Chem

edges, graph_idxs, graph_labels, link_labels, \
            node_labels, max_num_nodes = preprocess_graph_data('Mutagenicity')
            
graphs = get_graphs_from_data(edges, graph_idxs, graph_labels, \
                        link_labels, node_labels, max_num_nodes)
smiles = [graph_to_smiles(x['x'], x['adj_matrix'], x['edge_attr_matrix'], x['mask'], "Mutagenicity") for x in graphs]

mols = [Chem.MolFromSmiles(x) for x in smiles if Chem.MolFromSmiles(x) is not None]
idxs = [i for i,x in enumerate(smiles) if Chem.MolFromSmiles(x) is not None]
mols = pd.DataFrame(mols)
mols.to_csv(open("valid_smiles.csv", "w"), index= False, header =False)

graph_labels = [x for i,x in enumerate(graph_labels) if i in idxs]
graph_labels = pd.DataFrame(graph_labels)
graph_labels.to_csv(open("valid_idxs.csv", "w"), index= False, header =False)


filtered_smiles = [Chem.MolToSmiles(x) for x in mols if x.GetNumAtoms() <= 100]
idxs = [i for i, x in zip(idxs,mols) if x.GetNumAtoms() <= 100]

filtered_smiles = pd.DataFrame(filtered_smiles)
filtered_smiles.to_csv(open("smiles.csv", "w"), index= False, header =False)

graph_labels = [x for i,x in enumerate(graph_labels) if i in idxs]
graph_labels = pd.DataFrame(graph_labels)
graph_labels.to_csv(open("idxs.csv", "w"), index= False, header =False)
