from utils.datasets import Dataset
import numpy as np
import pandas as pd
import torch

datasets = ["AIDS", "Mutagenicity", "Tox21", "BBBP", "ClinTox"]
max_nodes_list = []
ave_nodes_list = []
ave_edges_list = []
num_graphs_list = []
for d in datasets:
    dataset = Dataset(d)
    max_nodes = dataset.max_num_nodes
    ave_nodes = np.mean([graph['mask'].sum() for graph in dataset.graphs])
    ave_edges = np.mean([graph['adj_matrix'].sum()/2 for graph in dataset.graphs])
    max_nodes_list.append(max_nodes)
    ave_nodes_list.append(ave_nodes)
    ave_edges_list.append(ave_edges)
    num_graphs_list.append(len(dataset.graphs))

res = pd.DataFrame({'dataset': datasets, 'max_nodes': max_nodes_list, 'ave_nodes': ave_nodes_list, 'ave_edges': ave_edges_list, 'number of graphs': num_graphs_list})
res.to_csv(open("dataset_info.csv", "w"))