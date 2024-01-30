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
    if(len(data[0]) == 1):
        processed_data = [float(item[0]) for item in data]
    else:
        processed_data = [[float(item) for item in row] for row in data]
    return processed_data

def convert_node_labels_map(map_to_transfer):
    inverted_node_label_map = {sym: label for label,sym in NODE_LABEL_MAP.items()}
    for key in map_to_transfer:
        map_to_transfer[key] = inverted_node_label_map[map_to_transfer[key]]
    return map_to_transfer

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

    if dataset.lower() == "aids":
        node_labels = [row[1] for row in node_labels]
    elif dataset.lower() == "mutagenicity":
        node_labels_map = {0.0: 'C', 1.0: 'O', 2.0: 'Cl', 3.0: 'H', 4.0: 'N', 5.0: 'F', 6.0: 'Br', 
                           7.0: 'S', 8.0: 'P', 9.0: 'I', 10.0: 'Na', 11.0: 'K', 12.0: 'Li', 13.0: 'Ca'}
        node_labels_map = convert_node_labels_map(node_labels_map)
        node_labels = [node_labels_map[node_label] for node_label in node_labels]

    return edges, graph_idxs, graph_labels, link_labels, node_labels, max_num_nodes

def preprocess_smiles_data(dataset):
    smiles = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_smiles.csv')
    graph_labels = read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_graph_labels.csv')
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

        mask = torch.BoolTensor([True] * len(graph_nodes_idxs) + [False] * (max_num_nodes - len(graph_nodes_idxs)))
        
        graphs.append({"x" : node_attrs, "adj_matrix": adj_matrix, "edge_attr_matrix": edge_attr_matrix, \
                       'graph_label': graph_labels[graph_idx], 'mask': mask, 'num_nodes': len(graph_nodes_idxs)})
    return graphs

def get_graphs_from_smiles(smiles, graph_labels):
    graphs = []
    for idx, smile_str in enumerate(smiles):
        node_attrs, adj_matrix, edge_attr_matrix, mask = smiles_to_graph(smile_str)
        graphs.append({"x" : node_attrs, "adj_matrix": adj_matrix, "edge_attr_matrix": edge_attr_matrix, \
                       'graph_label': graph_labels[idx], 'mask': mask, 'num_nodes': torch.sum(mask).item()})

def get_text_attrs(graphs, dataset):
    if not os.path.isfile(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_output.csv'):
        smiles_list = [graph_to_smiles(graph['x'], graph['adj_matrix'], graph['edge_attr_matrix'], \
                                       graph['mask']) for graph in graphs]
        writer = csv.writer(open(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_output.csv', 'w'))
        for smile in tqdm(smiles_list):
            message, completion, prompt = get_description(smile, dataset)
            writer.writerow([smile, message, completion, prompt])
    
    return pd.read_csv(f'{DATASET_ROOT_PATH}{dataset}/{dataset}_output.csv', 
                       header=None, names=['SMILES', 'captions', 'prompt_lens', 'answer_lens'])

def get_description(molecule_data, dataset):
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": query_format.format(molecule_data = molecule_data, dataset_description = DATASET_QUERY_MAP[dataset][0]),
            },
        ],
        temperature=0.3
    )
    return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens
