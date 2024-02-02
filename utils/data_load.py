import os
import openai
import csv
import pandas as pd
from tqdm import tqdm
from collections import Counter
from itertools import groupby

from utils.smiles import *
from utils.pre_defined import *

DATASET_ROOT_PATH = "./data/"
openai.api_key = openai_api_key

def read_csv(file_path):
    file = open(file_path, "r")
    data = list(csv.reader(file, delimiter=","))
    try:
        if(len(data[0]) == 1):
            processed_data = [float(item[0]) for item in data]
        else:
            processed_data = [[float(item) for item in row] for row in data]
    except ValueError: # in this case, it's a SMILES string
        processed_data = [item[0] for item in data]
    return processed_data

'''
@edges: list of edges in graph (node indices start at 0)
@graph_idx: list of graph indices for all nodes (graph index starts at 0)
@graph_labels: list of graph labels for all graphs (ranges from 1 - 3)
@link_labels: list of link labels for all edges
@node_labels: atom type (ranges from 0 - 38)
'''
def preprocess_graph_data(dataset):
    edges = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}.edges')
    graph_idxs = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}.graph_idx')
    graph_labels = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}.graph_labels')
    link_labels = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}.link_labels')
    node_labels = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}.node_labels')
    
    # finds the most frequent graph index --> contains the most nodes
    count = Counter(graph_idxs)
    _, max_num_nodes = count.most_common(1)[0]
    edges = [[node - 1 for node in edge] for edge in edges]
    graph_idxs = [idx - 1 for idx in graph_idxs]
    link_labels = [label + 1 for label in link_labels]
    graph_labels = [int(not bool(label)) for label in graph_labels] # inverse graph labels (1 = desired)

    if dataset == "AIDS":
        node_labels = [row[1] for row in node_labels]
    elif dataset.lower() == "mutagenicity":
        node_labels = [node_label + 1 for node_label in node_labels]

    return edges, graph_idxs, graph_labels, link_labels, node_labels, max_num_nodes

def preprocess_smiles_data(dataset):
    smiles = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_smiles.csv')
    graph_labels = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_graph_labels.csv')
    graph_labels = [int(label) for label in graph_labels]
        
    return smiles, graph_labels


def get_graphs_from_data(edges, graph_idxs, graph_labels, link_labels, node_labels, max_num_nodes):
    graphs = []
    graph_node_idxs_list = [list(g) for _,g in groupby(range(len(graph_idxs)),lambda idx:graph_idxs[idx])] # groups the node indices in each graph
    node_to_graph_map = {node_idx: graph_idx for node_idx, graph_idx in enumerate(graph_idxs)}
    graph_edge_idxs_list = [list(g) for _,g in groupby(range(len(edges)),lambda idx:node_to_graph_map[edges[idx][0]])]

    for graph_idx, (graph_edge_idxs, graph_nodes_idxs) in enumerate(zip(graph_edge_idxs_list, graph_node_idxs_list)):
        start_index_diff = graph_nodes_idxs[0]
        adj_matrix = torch.zeros((max_num_nodes, max_num_nodes), dtype=torch.float)
        edge_attr_matrix = torch.zeros((max_num_nodes, max_num_nodes), dtype=torch.float)
        
        for edge_idx in graph_edge_idxs:
            i = int(edges[edge_idx][0]) - start_index_diff
            j = int(edges[edge_idx][1]) - start_index_diff
            adj_matrix[i,j] = 1
            adj_matrix[j,i] = 1
            edge_attr_matrix[i,j] = link_labels[edge_idx]
            edge_attr_matrix[j,i] = link_labels[edge_idx]

        node_attrs = [[node_labels[node_idx]] for node_idx in graph_nodes_idxs]
        node_attrs.extend([[0] for _ in range(max_num_nodes - len(graph_nodes_idxs))])
        node_attrs = torch.Tensor(node_attrs)

        mask = torch.BoolTensor([True] * len(graph_nodes_idxs) + [False] * (max_num_nodes - len(graph_nodes_idxs)))
        
        graphs.append({"x" : node_attrs, "adj_matrix": adj_matrix, "edge_attr_matrix": edge_attr_matrix, \
                       'graph_label': graph_labels[graph_idx], 'mask': mask, 'num_nodes': len(graph_nodes_idxs)})
    return graphs

def get_graphs_from_smiles(smiles, graph_labels, dataset):
    graphs = []
    max_nodes = 0
    invalid_smiles = []
    for idx, smile_str in enumerate(smiles):
        node_attrs, adj_matrix, edge_attr_matrix, mask = smiles_to_graph(smile_str, dataset, add_hydrogen=True)
        
        if node_attrs == None:
            invalid_smiles.append(idx)
            continue

        if len(node_attrs) > max_nodes:
            max_nodes = len(node_attrs)
        graphs.append({"x" : node_attrs, "adj_matrix": adj_matrix, "edge_attr_matrix": edge_attr_matrix, \
                       'graph_label': graph_labels[idx], 'mask': mask, 'num_nodes': torch.sum(mask).item()})
    for idx in range(len(graphs)):
        cur_num_nodes = len(graphs[idx]['x'])
        graphs[idx]['x'] = torch.cat((graphs[idx]['x'], torch.Tensor([[0] for _ in range (max_nodes - cur_num_nodes)]))) 
        
        adj_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
        edge_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
        adj_matrix[0:cur_num_nodes, 0:cur_num_nodes] = graphs[idx]['adj_matrix']
        edge_matrix[0:cur_num_nodes, 0:cur_num_nodes] = graphs[idx]['edge_attr_matrix']
        graphs[idx]['adj_matrix'] = adj_matrix
        graphs[idx]['edge_attr_matrix'] = edge_matrix

        graphs[idx]['mask'] = torch.cat((graphs[idx]['mask'], torch.BoolTensor([False] * (max_nodes - cur_num_nodes))))
    
    if len(invalid_smiles) > 0:
        smiles = [smiles[i] for i in range(len(smiles)) if i not in invalid_smiles]
        graph_labels = [graph_labels[i] for i in range(len(graph_labels)) if i not in invalid_smiles]
    return graphs, max_nodes, smiles, graph_labels

def get_text_attrs(graphs, dataset, smiles_list=None):
    if not os.path.isfile(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_output.csv'):
        if smiles_list == None:
            smiles_list = [graph_to_smiles(graph['x'], graph['adj_matrix'], graph['edge_attr_matrix'], \
                                           graph['mask'], dataset) for graph in graphs]
        writer = csv.writer(open(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_output.csv', 'w'))
        for smile in tqdm(smiles_list):
            message, completion, prompt = get_description(smile, dataset)
            writer.writerow([smile, message, completion, prompt])
    
    return pd.read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_output.csv', 
                       header=None, names=['SMILES', 'captions', 'prompt_lens', 'answer_lens'])

def get_description(molecule_data, dataset):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "user",
                "content": query_format.format(molecule_data = molecule_data, dataset_description = DATASET_QUERY_MAP[dataset][0]),
            },
        ],
        temperature=0.3
    )
    return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens

# def get_description_batch(molecule_list, dataset):
#     prompt = ""
#     for mol in molecule_list:
#         prompt += mol + '\n'

#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo-1106",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f'The following are molecules in SMILES representation. Please generate a text description for every molecule STRICTLY in the form of: "This molecule contains __, __, __, and __ functional groups, in which __ may be the most influential for {DATASET_QUERY_MAP[dataset][0]}." NO OTHER sentence patterns allowed. Here, __ is the functional groups (best each less than 10 atoms) or significant subgraphs alphabetically. If you can not find 4 functional groups significant subgraphs, you can just put all you have found in the __ areas)'
#             },
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ],
#         temperature=0.3
#     )
#     return response.choices[0].message.content
