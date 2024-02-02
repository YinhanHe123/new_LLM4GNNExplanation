import torch
import numpy as np
from rdkit import Chem

def evaluate_gce_model(cf_model, gcn, dataset):
    """
    Evaluate the GCE model.
    """
    validity, valid_cf_list = compute_validity(cf_model, gcn)
    proximity = compute_proximity(valid_cf_list, dataset)
    return validity, proximity


def compute_validity(cf_list, gt_gnn):
    cf_preds = []
    valid_cf_list = []
    for cf in cf_list:
        if cf['x'] is not None:
            cf['x'] = cf['x'].unsqueeze(0)
            cf['adj'] = cf['adj'].unsqueeze(0)
            cf['edge_attr'] = cf['edge_attr'].unsqueeze(0)
            cf_pred = gt_gnn(cf)['y_pred'].argmax(dim=1).item()
            cf_preds.append(cf_pred)
            valid_cf_list.append(cf)
        else:
            cf_preds.append(0)
    percent_correct = sum(cf_preds) / len(cf_preds) * 100
    return percent_correct, valid_cf_list 

def compute_proximity(valid_cf_list, dataset):
    dist = 0
    for cf in valid_cf_list:
        torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2)))
        dist += torch.sqrt(torch.sum((dataset.__getitem__(cf['graph_idx'])['adj'] - cf['adj']).pow(2))).item()
    return dist / len(valid_cf_list)