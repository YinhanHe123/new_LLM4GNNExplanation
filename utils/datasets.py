import numpy as np
import torch
from utils.data_load import *
from utils.smiles import *
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import re

class Dataset(BaseDataset):
    def __init__(self, dataset, generate_text = True):
        self.dataset = dataset
        if dataset in ['AIDS', 'Mutagenicity']:
            # load graph data
            edges, graph_idxs, self.graph_labels, link_labels, \
            node_labels, self.max_num_nodes = preprocess_graph_data(dataset)
            
            # converts graph data into a list of maps with the keys: "x", "adj_matrix", "edge_attr_matrix", 'graph_label', 'mask'
            self.graphs = get_graphs_from_data(edges, graph_idxs, self.graph_labels, \
                                    link_labels, node_labels, self.max_num_nodes)
            if generate_text:
                # load text data
                data_csv = get_text_attrs(self.graphs, dataset)
                self.text_attrs = data_csv['captions'].values.tolist()
                self.smiles = data_csv['SMILES'].values.tolist()
            else:
                self.text_attrs = None
                self.smiles = [graph_to_smiles(graph['x'], graph['adj_matrix'], graph['edge_attr_matrix'], \
                                               graph['mask']) for graph in self.graphs]
        elif dataset in ['BBBP', 'SIDER', 'Tox21']:
            self.smiles, self.graph_labels = preprocess_smiles_data(dataset)
            self.graphs, self.max_num_nodes = get_graphs_from_smiles(self.smiles, self.graph_labels)
            
            if generate_text:
                data_csv = get_text_attrs(self.graphs, dataset, self.smiles)
                self.text_attrs = data_csv['captions'].values.tolist()
            else:
                self.text_attrs = None
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.graphs)
    
    def get_key_conponent(self, caption):
        key_component = re.search(r'(in which |where |of which )(the )*(.*?) may', caption, re.IGNORECASE)
        key_component = key_component.group(3) if key_component else None
        assert key_component is not None
        
        pattern = re.compile(re.escape(key_component), re.IGNORECASE)
        caption_to_be_revised = pattern.sub('__', caption)
        
        return key_component, caption_to_be_revised
    
    def get_negative_idxs(self):
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
                likely='likely' if graph['graph_label'] == 0 else 'unlikely',
                dataset_description = DATASET_QUERY_MAP[self.dataset][0],
                molecule_description = DATASET_QUERY_MAP[self.dataset][1],
            ),
            'num_atoms': graph['num_nodes'],
            'text': self.text_attrs[graph_idx],
            'label': torch.tensor(graph['graph_label'], dtype=torch.long),
            'graph_idx': torch.tensor(graph_idx, dtype=torch.int)
        }