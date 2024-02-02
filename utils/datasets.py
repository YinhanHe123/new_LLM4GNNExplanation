import math
import numpy as np
import torch
from utils.data_load import *
from utils.smiles import *
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import re
import contextlib


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def sample_graphs_by_label(graphs, dataset, sample_size=2000):
    if dataset == 'Mutagenicity':
        graph_labels = [graph['graph_label'] for graph in graphs]
        # get indices of graph_labels that are 0 and 1
        neg_idxs, pos_idxs = np.where(np.array(graph_labels) == 0)[0], \
                            np.where(np.array(graph_labels) == 1)[0]
        neg_ratio, pos_ratio = len(neg_idxs)/len(graph_labels), len(pos_idxs)/len(graph_labels)
        print(f"Negative ratio: {neg_ratio}, Positive ratio: {pos_ratio}")
        # sample (without putback) 2000 graphs proportionally by their labels
        with temp_seed(42):
            neg_sampled_idxs = np.random.choice(neg_idxs, size=round(sample_size*neg_ratio)+1, replace=False)
            pos_sampled_idxs = np.random.choice(pos_idxs, size=round(sample_size*pos_ratio), replace=False)
        neg_sampled_idxs = np.delete(neg_sampled_idxs, np.argwhere(neg_sampled_idxs==2143))
        sampled_idxs = np.concatenate([neg_sampled_idxs, pos_sampled_idxs])
        # get the sampled graphs
        sampled_graphs = [graphs[idx] for idx in sampled_idxs]
        return sampled_graphs, sampled_idxs, neg_sampled_idxs
    elif dataset in ['Tox21', 'BBBP']:
        graph_labels = [graph['graph_label'] for graph in graphs]
        neg_idxs, pos_idxs = np.where(np.array(graph_labels) == 0)[0], \
                            np.where(np.array(graph_labels) == 1)[0]
        neg_ratio, pos_ratio = len(neg_idxs)/len(graph_labels), len(pos_idxs)/len(graph_labels)
        print(f"Negative ratio: {neg_ratio}, Positive ratio: {pos_ratio}")
        with temp_seed(42):
            neg_sampled_idxs = np.random.choice(neg_idxs, size=round(sample_size*neg_ratio), replace=False)
            pos_sampled_idxs = np.random.choice(pos_idxs, size=round(sample_size*pos_ratio), replace=False)
        sampled_idxs = np.concatenate([neg_sampled_idxs, pos_sampled_idxs])
        sampled_graphs = [graphs[idx] for idx in sampled_idxs]
        return sampled_graphs, sampled_idxs, neg_sampled_idxs
    else:
        raise NotImplementedError

class Dataset(BaseDataset):
    # ALERT!! Remember to delete Mutagenicity and Tox21's csv before running, but all the other csvs should be REMAINED (Feb 2, 1:45 a.m., 2024)!!!
    # Once a full run of the code is done, the csv file of Muta. and Tox21 can be remained. 
    # the get_text_attrs() will 
    # (1) if csv file existed: read the existed csv file (either the number of graphs are smaller than 2000 or larger than 2000)
    # (2) if csv file not existed: generate the text attributes (for the sampled graphs i put in the input of the function) and save them into a csv file.
    def __init__(self, dataset, generate_text=True):
        self.dataset = dataset
        self.negative_idxs = None
        print("----------------------Loading data----------------------\n")
        if dataset in ['AIDS', 'Mutagenicity']:
            # load graph data
            edges, graph_idxs, self.graph_labels, link_labels, \
            node_labels, self.max_num_nodes = preprocess_graph_data(dataset)
            
            # converts graph data into a list of maps with the keys: "x", "adj_matrix", "edge_attr_matrix", 'graph_label', 'mask'
            print("----------------------Generating graphs----------------------\n")
            self.graphs = get_graphs_from_data(edges, graph_idxs, self.graph_labels, \
                                    link_labels, node_labels, self.max_num_nodes)
            if len(self.graphs) > 2000:
                self.graphs, _, self.negative_idxs = sample_graphs_by_label(self.graphs, dataset)
            print("----------------------Getting text attributes----------------------\n")
            data_csv = get_text_attrs(self.graphs, dataset)
            self.text_attrs = data_csv['captions'].values.tolist()
            self.smiles = data_csv['SMILES'].values.tolist() 

            if len(self.text_attrs) > len(self.graphs):
                self.text_attrs = [self.text_attrs[i] for i in sampled_idxs]
                self.smiles = [self.smiles[i] for i in sampled_idxs]
        
        elif dataset in ['BBBP', 'SIDER', 'Tox21', 'ClinTox']:
            self.smiles, self.graph_labels = preprocess_smiles_data(dataset)
            self.graphs, self.max_num_nodes, _, _ = \
                get_graphs_from_smiles(self.smiles, self.graph_labels, self.dataset)
            if len(self.graphs) > 2000:
                self.graphs, sampled_idxs, self.negative_idxs = sample_graphs_by_label(self.graphs, dataset)
            print("----------------------Getting text attributes----------------------\n")
            data_csv = get_text_attrs(self.graphs, dataset)
            self.text_attrs = data_csv['captions'].values.tolist()
            if len(self.text_attrs) > len(self.graphs):
                self.text_attrs = [self.text_attrs[i] for i in sampled_idxs]
                self.smiles = [self.smiles[i] for i in sampled_idxs]
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.graphs)
    
    def get_key_conponent(self, caption):
        key_component = re.search(r'(in which |where |of which )(the )*(.*?) may', caption, re.IGNORECASE)
        key_component = key_component.group(3) if key_component else None
        if(key_component == None and len(caption.split(",")) == 2): # only contains 1 functional group
            key_component = re.search(r'This molecule contains (a |an |the )*(.*?) (which may|and)', caption, re.IGNORECASE)
            key_component = key_component.group(2) if key_component else None
        assert key_component is not None
        pattern = re.compile(re.escape(key_component), re.IGNORECASE)
        caption_to_be_revised = pattern.sub('__', caption)
        
        return key_component, caption_to_be_revised
    
    def get_negative_idxs(self):
        if self.negative_idxs is not None:
            return self.negative_idxs
        labels_arr = np.array(self.graph_labels)
        indices = np.where(labels_arr == 0)[0]
        return indices
    
    def get_dataloaders(self, batch_size, data_split=[0.5, 0.25, 0.25], shuffle=True, num_workers=0, mask_pos = False):
        # return the list of train, val and test dataloaders according to data_split
        assert sum(data_split) == 1
        assert len(data_split) == 3
        if mask_pos:
            indices = self.get_negative_idxs().tolist()
            dataset_size = len(indices)
        else:
            dataset_size = len(self.graphs)
            indices = list(range(dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_split = int(np.floor(data_split[0] * dataset_size))
        val_split = int(np.floor(data_split[1] * dataset_size))
        train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:train_split+val_split], indices[train_split+val_split:]
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(self, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        return train_loader, val_loader, test_loader
    
    def __getitem__(self, graph_idx):
        graph = self.graphs[graph_idx]
        key_component, caption_to_be_revised = self.get_key_conponent(self.text_attrs[graph_idx])
        
        return {
            'x': graph['x'],
            'edge_attr': graph['edge_attr_matrix'],
            'adj': graph['adj_matrix'],
            'mask': graph['mask'],
            'smiles': self.smiles[graph_idx],
            'cf_query': cf_query_format.format(
                smiles=self.smiles[graph_idx],
                key_component=key_component,
                caption_to_be_revised=caption_to_be_revised,
                likely='increase',
                dataset_description = DATASET_QUERY_MAP[self.dataset][0],
                molecule_description = DATASET_QUERY_MAP[self.dataset][1],
            ),
            'num_atoms': graph['num_nodes'],
            'text': self.text_attrs[graph_idx],
            'label': torch.tensor(graph['graph_label'], dtype=torch.long),
            'graph_idx': torch.tensor(graph_idx, dtype=torch.int),
            'dataset': self.dataset
        }