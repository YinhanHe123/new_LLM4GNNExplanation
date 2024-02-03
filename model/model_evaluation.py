import torch
import numpy as np
from rdkit import Chem

def evaluate_gce_model(cf_list, gcn, dataset):
    """
    Evaluate the GCE model.
    """
    validity, valid_cf_list = compute_validity(cf_list, gcn)
    proximity = compute_proximity(valid_cf_list, dataset)
    return validity, proximity


def compute_validity(cf_list, gt_gnn):
    cf_preds = []
    valid_cf_list = []
    for cf in cf_list:
        if cf['x'] is not None:
            valid_cf_list.append(cf)
        cf_preds.append(cf['true_prob'] > 0.5)
    percent_correct = sum(cf_preds) / len(cf_preds) * 100
    return percent_correct, valid_cf_list 

def compute_proximity(valid_cf_list, dataset):
    dist = 0
    for cf in valid_cf_list:
        torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2)))
        dist += torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2))).item()
    if len(valid_cf_list) == 0:
        return 0
    return dist / len(valid_cf_list)
