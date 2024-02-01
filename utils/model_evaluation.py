import torch
import numpy as np
from rdkit import Chem

def evaluate_gce_model(cf_model, gcn):
    """
    Evaluate the GCE model.
    """
    validity, valid_cf_list = compute_validity(cf_model, gcn)
    proximity = compute_proximity(valid_cf_list, gcn)
    return validity, proximity


def compute_validity(cf_list, gt_gnn):
    for cf in cf_list:
        cf_preds = gt_gnn(cf).argmax(dim=1)
    percent_correct = torch.count_nonzero(cf_preds).item() / cf_preds.shape[-1] * 100

    valid_cf_list = []
    for cf in cf_list:
        temp_mol = Chem.MolFromSmiles(cf['smiles'])
        if temp_mol != None:
            valid_cf_list.append(cf)
    return percent_correct, valid_cf_list 

def compute_proximity(valid_cf_list, dataset):
    dist = 0
    for cf in valid_cf_list:
        dist += torch.cdist(dataset.__getitem__(cf['graph_idx'])['adj'], cf['adj'])
    return dist / len(valid_cf_list)